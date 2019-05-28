import torch
import torch.nn as nn
import torchvision as vision
import numpy as np
from constants.constants import NUM_CLASSES, ATTN_DIM, DECODER_DIM, ENCODER_DIM, MAX_LABEL_LEN, TRAIN_PROPORTION_PATH

class EncoderCNNwithHOG(nn.Module):
    """
        Module for encoder CNN, adapted from 
        https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py
    """
    def __init__(self, args, encoded_image_size=14):
        super(EncoderCNNwithHOG, self).__init__()
        self.sigmoid = nn.Sigmoid()
        
        if args.use_pretrained and (args.resnet_path is None):
            self.resnet50 = vision.models.resnet50(pretrained=True) # load ImageNet pretrained weights
        elif args.resnet_path is not None:
            # If we want to use already self-trained resnet50 including modified last FC layer
            self.resnet50 = vision.models.resnet50(pretrained=True)
            num_ftrs = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_ftrs, NUM_CLASSES)
            
            checkpoint_dict = torch.load(args.resnet_path)
            pretrained_dict = checkpoint_dict['model_state']
            model_dict = self.resnet50.state_dict()
            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # Overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # Load the new state dict
            self.resnet50.load_state_dict(model_dict)
            
        else:
            self.resnet50 = vision.models.resnet50(pretrained=False) # no pretraining
        
        # Freeze all or some layers
        if args.feature_extracting:
            self.set_parameter_requires_grad(self.resnet50, feature_extracting=True, nlayers_to_freeze=None)         
        else:
            print('Fine-tune ResNet50 with {} layers freezed...'.format(args.nlayers_to_freeze))
            self.set_parameter_requires_grad(self.resnet50, 
                                             feature_extracting=False,
                                             nlayers_to_freeze=args.nlayers_to_freeze)
            
        # if resnet_path is None, need to modify the last FC layer from original ResNet50
        if args.resnet_path is None:
            num_ftrs = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_ftrs, NUM_CLASSES)
            
        # Exclude linear and pool layers
        modules = list(self.resnet50.children())[:-2]
        self.conv_features = nn.Sequential(*modules) # all layers until last pool layer (inclusive)
 
        # Add HOG to last FC layer
        self.hogfc = nn.Linear(num_ftrs+10368, NUM_CLASSES)
    
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
    def forward(self, x, hog_features):
        """
        Args:
            x: image tensor, (batch_size, C, H, W)
            hog_features: hog features tensor
        Returns:
            v_prob: (batch_size, NUM_CLASSES)
            v_feat: (batch_size, encoded_image_size, encoded_image_size, 2048)
        """
        
        # Features from last conv layer
        conv_feat = self.conv_features(x) 
        conv_feat = self.adaptive_pool(conv_feat) # (batch_size, 2048, encoded_image_size, encoded_image_size)
        # batch size
        B = conv_feat.shape[0]
        # flatten resnet_features
        conv_feat = conv_feat.reshape(B,-1)
        # concatenate with hog feature
        # convert to (batch_size*6, hog_vector_size)
        hog_features = hog_features.reshape(B,-1)
        features_concat = torch.cat((conv_feat.float(), hog_features.float()),dim=1)
        # FC
        scores = self.hogfc(features_concat)
        # mlp        
        # scores = self.hogmlp(features_concat) 
        
        # For RNN        
        v_feat = conv_feat.permute(0, 2, 3, 1) # (batch_size, encoded_image_size, encoded_image_size, 2048)
        v_prob = self.sigmoid(scores) # (batch_size, NUM_CLASSES)
    
        return v_prob, v_feat
               
    def set_parameter_requires_grad(self, model, feature_extracting=False, nlayers_to_freeze=None):
        # freeze some layers
        if nlayers_to_freeze is not None:
            ct = 0
            for name, child in model.named_children():
                ct += 1
                if ct < nlayers_to_freeze:
                    for name2, params in child.named_parameters():
                        params.requires_grad = False
        else:
            # if feature extracting, freeze all layers
            if feature_extracting: 
                for param in model.parameters():
                    param.requires_grad = False
    

class Attn(nn.Module):
    """
        Compute the attention weighted context vector z
        Adapted from: https://github.com/parksunwoo/show_attend_and_tell_pytorch
    """
    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super(Attn, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attn_dim)
        self.decoder_att = nn.Linear(decoder_dim, attn_dim)
        self.full_att = nn.Linear(attn_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        z = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # (batch_size, encoder_dim) e.g. (96, 2048)
        return z, alpha

        
class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_dim, decoder_dim, encoder_dim, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.attn_dim = attn_dim
        self.decoder_dim = decoder_dim
        
        # Attention layer
        self.attn = Attn(encoder_dim, decoder_dim, attn_dim)

        self.dropout = nn.Dropout(p=dropout)
        
        # Decoder / prediction layer
        self.decode_step = nn.LSTMCell(NUM_CLASSES * 2 + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim) # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(decoder_dim, decoder_dim)
        self.fc2 = nn.Linear(decoder_dim, NUM_CLASSES)
        
        # Initialize weights
        self.init_weights()
        
        # Load class proportions
        with open(TRAIN_PROPORTION_PATH, 'rb') as f:
            self.class_prop = np.loadtxt(f, delimiter=',', skiprows=1)
            self.class_prop = self.class_prop[:,1]

    def init_weights(self):
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)


    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out) # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, v_prob, v_feat, device, max_label_len=10, 
                prob_path_thresh=1e-6, loss_fn=None, labels=None, 
                is_eval=False, beam_search=False, beam_size=1):
        """
        Args:
            v_prob: predicted probability from CNN, (batch_size, NUM_CLASSES)
            v_feat: visual map features extracted from CNN
            device: device
            max_label_len: maximum label length in train split
            prob_path_thresh: prediction path probability threshold to terminate test prediction
            loss_fn: loss function to compute loss
            labels: one-hot encoded ground truth labels, (batch_size, ncrops, NUM_CLASSES)
            beam_search: True if using beam search, for evaluation only (not training)
        Returns:
            hard_labels: Predicted hard labels in one-hot, shape (batch_size, NUM_CLASSES)
            masked_alphas: Alphas (attention) scores, shape (batch_size, max_label_len, 14*14)
            total_loss: None if labels is none
        """
        
        if is_eval and beam_search:
            hard_labels, total_loss, masked_alphas = self.beam_search(v_prob, v_feat, 
                                                                      device, max_label_len,
                                                                      prob_path_thresh,
                                                                      labels, beam_size)
            
        else: 
            #No beam search
            batch_size = v_feat.size(0)
            encoder_dim = v_feat.size(-1)
        
            v_feat = v_feat.view(batch_size, -1, encoder_dim) # (batch_size, num_pixels, encoder_dim)
            num_pixels = v_feat.size(1)
                
            # Initialize LSTM state
            h, c = self.init_hidden_state(v_feat)  # (batch_size, decoder_dim)
        
            # Initialize v_pred
            v_pred = v_prob # (batch_size, NUM_CLASSES)
        
            # Initialize candidate pool
            C = np.ones((batch_size, NUM_CLASSES)) # (batch_size, NUM_CLASSES)
        
            y_tilt = torch.zeros(batch_size, NUM_CLASSES).to(device)
            y_tilt_num = np.zeros((batch_size, max_label_len))
            alphas = np.zeros((batch_size, max_label_len, num_pixels)) # for visualization purpose
            hard_labels = np.zeros((batch_size, max_label_len))
            soft_probs = np.zeros((batch_size, max_label_len))
            loss = 0.    
            for t in range(max_label_len):
                # Pass through attention layer
                z, alpha = self.attn(v_feat, h)
                gate = self.sigmoid(self.f_beta(h))
                z = gate * z
                # Pass through prediction layer
                h, c = self.decode_step(torch.cat([v_pred, z, y_tilt], dim=1), (h, c))              
                fc1_out = self.fc1(self.dropout(h))
                logit = self.fc2(self.dropout(fc1_out)) # (batch_size, NUM_CLASSES)
                probs = self.sigmoid(logit) # (batch_size, NUM_CLASSES)
                alphas[:,t,:] = alpha.detach().cpu().numpy()

                # Only choose from the candidate pool to prevent duplicated labels
                ps = probs.detach().cpu().numpy() # (batch_size, NUM_CLASSES)
                ps *= C # mask the probabilities with the candidate pool
                l = np.argmax(ps, axis=1) # current time step hard label, shape (batch_size,)
            
                prob_max = np.asarray([ps[idx,l[idx]] for idx in range(batch_size)])
                for idx in range(batch_size):
                    y_tilt[idx,int(l[idx])]=1
                    C[idx,int(l[idx])] = 0. # remove newly predicted label from candidate pool
                    
                y_tilt_num[:,t] = l
                
            
                soft_probs[:,t] = prob_max
            
                if labels is not None and loss_fn is not None:
                    # Loss sum over all time steps
                    # Note that in the original paper, it performs GD at each time step
                    # TODO: Can also try SGD at each time step
                    loss += loss_fn(logit, labels.view(-1, NUM_CLASSES))
                
                v_pred = probs
        
            if labels is not None:
                total_loss = loss / max_label_len
            else:
                total_loss = None
            
        
            # If test, discard labels after the time step when path probability is lower than the threshold
            if is_eval:
                prob_path = np.ones((batch_size, max_label_len))
                hard_labels = np.zeros((batch_size, NUM_CLASSES))
                thresholds = np.zeros((batch_size, max_label_len))
                for t in range(max_label_len):
                    # Compute normalized path probability, so that it is invariant to path length
                    #prob_path[:,t] = np.mean(np.log(soft_probs[:,:t]), axis=1)
                    prob_path[:,t] = np.prod(soft_probs[:,:t+1], axis=1)**(1/(t+1))
                    
                    # threshold to compare to
                    l_t = y_tilt_num[:,:t+1].astype(int) # (batch_size, t)                    
                    curr_class_prop = self.class_prop[l_t] # (batch_size, t)
                    #print(curr_class_prop)
                    thresholds[:,t] = np.prod(curr_class_prop * prob_path_thresh, axis=1)**(1/(t+1))
            
#                print(prob_path)
                #masks = prob_path >= prob_path_thresh
                # Use different thresholds for different classes
                #print(thresholds)
                masks = prob_path >= thresholds

                masked_alphas = alphas * masks[:,:,np.newaxis]
                for i in range(batch_size):
                    curr_labels = y_tilt_num[i, masks[i,:]].astype(int)
                    hard_labels[i, curr_labels] = 1
            else:
                hard_labels = y_tilt.detach().cpu().numpy()
                masked_alphas = alphas
        return hard_labels, total_loss, masked_alphas

    def beam_search(self, v_prob, v_feat, device, max_label_len=10,
                    prob_path_thresh=1e-3, labels=None, beam_size=3):
        """
        Args:
            v_prob: predicted probability from CNN, (batch_size, NUM_CLASSES)
            v_feat: visual map features extracted from CNN
            device: device
            max_label_len: maximum label length in train split
            prob_path_thresh: prediction path probability threshold to terminate test prediction
            labels: one-hot encoded ground truth labels, (batch_size, ncrops, NUM_CLASSES)
            beam_size: Number of beams to search
        Returns:
            hard_labels: Predicted hard labels in one-hot, shape (batch_size, NUM_CLASSES)
            masked_alphas: Alphas (attention) scores, shape (batch_size, max_label_len, 14*14)
            total_loss: None if labels is none
        """
        # Refer to above non-beam search implementation and 
        # https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py
        # Ideally the beam search should also be operated in batch, otherwise might be too slow           
        batch_size = v_feat.size(0)
        enc_image_size = v_feat.size(1)
        encoder_dim = v_feat.size(-1)
        v_feat = v_feat.view(batch_size, -1, encoder_dim) # (batch_size, num_pixels, encoder_dim)
        num_pixels = v_feat.size(1)
        k = beam_size
        # Start decoding
        step = 0
        # Initialize LSTM state
        h, c = self.init_hidden_state(v_feat)  # (batch_size, decoder_dim)
        # Initialize v_pred
        v_pred = v_prob # (batch_size, NUM_CLASSES)
        # Initialize candidate pool
        C = np.ones((batch_size, k, max_label_len, NUM_CLASSES)) # (batch_size, NUM_CLASSES)       
        y_tilt = torch.zeros(batch_size, NUM_CLASSES).to(device)
        y_tilt_num = np.zeros((batch_size, max_label_len))
        alphas = np.zeros((batch_size, max_label_len, num_pixels)) # for visualization purpose
        soft_probs = np.zeros((batch_size, k, max_label_len))
        thresholds = np.zeros((batch_size, max_label_len))
        
        while True:
            if step == 0:
                z, alpha = self.attn(v_feat, h)       
                gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
                z = gate * z
                h, c = self.decode_step(torch.cat([v_pred, z, y_tilt], dim=1), (h, c))
                fc1_out = self.fc1(self.dropout(h))
                logit = self.fc2(self.dropout(fc1_out)) # (batch_size, NUM_CLASSES)
                probs = self.sigmoid(logit) # (batch_size, NUM_CLASSES)
                alphas[:,step,:] = alpha.detach().cpu().numpy()
                # Only choose from the candidate pool to prevent duplicated labels
                ps = np.zeros((batch_size, k, max_label_len, NUM_CLASSES))
                l = np.zeros((batch_size, k))
                y_tilt_beam = torch.zeros(batch_size, k, max_label_len, NUM_CLASSES)
                y_tilt_num_beam = np.zeros((batch_size, k, max_label_len, max_label_len))
                v_pred_beam = torch.zeros(batch_size, k, max_label_len, NUM_CLASSES)
                h_beam = torch.zeros(batch_size, k, max_label_len, h.shape[1])
                c_beam = torch.zeros(batch_size, k, max_label_len, c.shape[1])
                alphas_beam = np.zeros((batch_size, k, max_label_len, num_pixels))
                for ik in range(k):
                    v_pred_beam[:,ik,step,:] = probs
                    h_beam[:,ik,step,:] = h
                    c_beam[:,ik,step,:] = c
                    alphas_beam[:,ik,step,:] = alphas[:,step,:]
                    ps[:,ik,step,:] = probs.detach().cpu().numpy()
                ps[:,:,step,:] *= C[:,:,step,:] # mask the probabilities with the candidate pool
                for ik in range(k):
                    for idx in range(batch_size):
                        l[idx,ik] = np.argsort(ps[idx,ik,step,:])[NUM_CLASSES-ik-1]
                        y_tilt[idx,int(l[idx,ik])] = 1
                        y_tilt_num[idx,step] = int(l[idx,ik])
                        C[idx,ik,step,int(l[idx,ik])] = 0
                    prob_max = np.asarray([ps[idx,ik,step,int(l[idx,ik])] for idx in range(batch_size)])
                    y_tilt_beam[:,ik,step,:] = y_tilt
                    y_tilt_num_beam[:,ik,step,:] = y_tilt_num
                    soft_probs[:,ik,step] = prob_max
                l_t = y_tilt_num_beam[:,0,step,:step+1].astype(int) # (batch_size, t)
                curr_class_prop = self.class_prop[l_t] # (batch_size, t)
                thresholds[:,step] = np.prod(curr_class_prop * prob_path_thresh, axis=1)**(1/(step+1))

            else:
                for ik in range(k):
                    h = h_beam[:,ik,step-1,:].to(device)
                    c = c_beam[:,ik,step-1,:].to(device)
                    z, alpha = self.attn(v_feat, h)                  
                    gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
                    z = gate * z                      
                    v_pred = v_pred_beam[:,ik,step-1,:].to(device)
                    y_tilt = y_tilt_beam[:,ik,step-1,:].to(device)
                    h, c = self.decode_step(torch.cat([v_pred, z, y_tilt], dim=1), (h, c))
                    fc1_out = self.fc1(self.dropout(h))
                    logit = self.fc2(self.dropout(fc1_out)) # (batch_size, NUM_CLASSES)
                    probs = self.sigmoid(logit) # (batch_size, NUM_CLASSES)          
                    alphas_beam[:,ik,step,:] = alpha.detach().cpu().numpy()
                    v_pred_beam[:,ik,step-1,:] = probs
                    h_beam[:,ik,step-1,:] = h
                    c_beam[:,ik,step-1,:] = c
                    ps[:,ik,step-1,:] = probs.detach().cpu().numpy()
                    for ic in range(NUM_CLASSES):
                        ps[:,ik,step-1,ic] *= soft_probs[:,ik,step-1]
                ps[:,:,step-1,:] *= C[:,:,step-1,:]
                for idx in range(batch_size):
                    ps_temp = ps[idx,:,step-1,:].reshape((-1,))
                    y_tilt_temp = y_tilt_beam[idx,:,:]
                    for ik in range(k):
                        l[idx,ik] = np.argsort(ps_temp)[ps_temp.shape[0]-ik-1]
                        prev_inds = int(l[idx,ik]) // NUM_CLASSES
                        next_inds = int(l[idx,ik]) % NUM_CLASSES
                        v_pred_beam[idx,ik,step,:] = v_pred_beam[idx,prev_inds,step-1,:]
                        h_beam[idx,ik,step,:] = h_beam[idx,prev_inds,step-1,:]
                        c_beam[idx,ik,step,:] = c_beam[idx,prev_inds,step-1,:]   
                        y_tilt_beam[idx,ik,step,:] =  y_tilt_beam[idx,prev_inds,step-1,:]
                        y_tilt_beam[idx,ik,step,next_inds] = 1                        
                        y_tilt_num_beam[idx,ik,step,:] = y_tilt_num_beam[idx,prev_inds,step-1,:]
                        y_tilt_num_beam[idx,ik,step,step] = next_inds
                        C[idx,ik,step,:] = C[idx,prev_inds,step-1,:] 
                        C[idx,ik,step,next_inds] = 0
                        ps[idx,ik,step,:] = ps[idx,prev_inds,step-1,:]
                        prob_max = ps_temp[int(l[idx,ik])]
                        soft_probs[idx,ik,step] = prob_max
                        alphas[idx,step,:] = alphas_beam[idx,prev_inds,step,:]
                l_t = y_tilt_num_beam[:,0,step,:step+1].astype(int) # (batch_size, t)
                curr_class_prop = self.class_prop[l_t] # (batch_size, t)
                thresholds[:,step] = np.prod(curr_class_prop * prob_path_thresh, axis=1)**(1/(step+1))
                #print(thresholds[0,:])
                    #print(y_tilt_num_beam[idx,:,:,:])
                    #print(soft_probs[idx,:,:])                    
            step += 1
            # Break if things have been going on too long
            if step >= max_label_len:
                break
        prob_path = np.ones((batch_size, max_label_len))
        hard_labels = np.zeros((batch_size, NUM_CLASSES))
        for t in range(max_label_len):
            prob_path[:,t] = soft_probs[:,0,t]**(1/(t+1)) 
        masks = prob_path >= thresholds
        #print(thresholds)
        masked_alphas = alphas * masks[:,:,np.newaxis]
        y_tilt_num = y_tilt_num_beam[:,0,max_label_len-1,:]
        for idx in range(batch_size):
            curr_labels = y_tilt_num[idx, masks[idx,:]].astype(int)
            hard_labels[idx, curr_labels] = 1
        total_loss = None
        return hard_labels, total_loss, masked_alphas

    
class HOG_CNN_RNN(nn.Module):
    """
        Wrapper class to combine CNN with visual attention RNN
    """
    def __init__(self, args, device):
        super(HOG_CNN_RNN, self).__init__()
        self.args = args
        self.device = device
        self.encoder = EncoderCNNwithHOG(args)
        self.decoder = AttnDecoderRNN(ATTN_DIM, DECODER_DIM, ENCODER_DIM, dropout=args.dropout)
    
    def forward(self, x, hog_features, labels=None, loss_fn=None, is_eval=False, test_only=False):
        v_prob, v_feat = self.encoder(x, hog_features)
        if test_only:
            beam_search = self.args.beam_search
        else:
            beam_search = False
        hard_labels, total_loss, alphas = self.decoder(v_prob, 
                                                       v_feat, 
                                                       self.device, 
                                                       max_label_len=MAX_LABEL_LEN, 
                                                       prob_path_thresh=self.args.prob_path_thresh, 
                                                       loss_fn=loss_fn,
                                                       labels=labels, 
                                                       is_eval=is_eval,
                                                       beam_search=beam_search,
                                                       beam_size=self.args.beam_size)

        return total_loss, hard_labels, alphas
        
        
        
        
    
    
    