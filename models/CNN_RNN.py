import torch
import torch.nn as nn
import torchvision as vision
import numpy as np
from constants.constants import NUM_CLASSES, ATTN_DIM, DECODER_DIM, ENCODER_DIM, MAX_LABEL_LEN

class EncoderCNN(nn.Module):
    """
        Module for encoder CNN, adapted from 
        https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py
    """
    def __init__(self, args, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.sigmoid = nn.Sigmoid()
        
        if args.use_pretrained:
            self.resnet50 = vision.models.resnet50(pretrained=True)
        else:
            self.resnet50 = vision.models.resnet50(pretrained=False)
            self.resnet50.load_state_dict(torch.load(args.load_path))
        
        if args.feature_extracting:
            self.set_parameter_requires_grad(self.resnet50, feature_extracting=True, nlayers_to_freeze=None)
            num_ftrs = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_ftrs, NUM_CLASSES)
            
        else:
            print('Fine-tune ResNet50 with {} layers freezed...'.format(args.nlayers_to_freeze))
            self.set_parameter_requires_grad(self.resnet50, 
                                             feature_extracting=False,
                                             nlayers_to_freeze=args.nlayers_to_freeze)
            num_ftrs = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_ftrs, NUM_CLASSES)
            
        # Remove linear and pool layers
        modules = list(self.resnet50.children())[:-2]
        self.conv_features = nn.Sequential(*modules) # all layers until last conv layer
        
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
    def forward(self, x):
        """
        Args:
            x: image tensor, (batch_size, C, H, W)
        Returns:
            v_prob: (batch_size, NUM_CLASSES)
            v_feat: (batch_size, encoded_image_size, encoded_image_size, 2048)
        """           
        v_prob = self.sigmoid(self.resnet50(x)) # (batch_size, NUM_CLASSES)
        v_feat = self.conv_features(x) # (batch_size, 2048, image_size/32, image_size/32))
        v_feat = self.adaptive_pool(v_feat) # (batch_size, 2048, encoded_image_size, encoded_image_size)
        v_feat = v_feat.permute(0, 2, 3, 1) # (batch_size, encoded_image_size, encoded_image_size, 2048)
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
                is_eval=False, beam_search=False, beam_size=3):
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
            hard_labels, masked_alphas, total_loss = self.beam_search(v_prob, v_feat, 
                                                                       device, max_label_len=max_label_len,
                                                                       prob_path_thresh=prob_path_thresh,
                                                                       labels=labels, beam_size=beam_size)
            
        else: 
            # No beam search
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
                y_tilt[:,l] = 1 # add current time step hard prediction
                y_tilt_num[:,t] = l
                C[:,l] = 0. # remove newly predicted label from candidate pool
            
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
                for t in range(max_label_len):
                    # Compute normalized path probability, so that it is invariant to path length
                    #prob_path[:,t] = np.mean(np.log(soft_probs[:,:t+1]), axis=1)
                    prob_path[:,t] = np.prod(soft_probs[:,:t+1], axis=1)**(1/(t+1))
            
#                print(prob_path)
                masks = prob_path >= prob_path_thresh
            
                masked_alphas = alphas * masks[:,:,np.newaxis]
                for i in range(batch_size):
                    curr_labels = y_tilt_num[i, masks[i,:]].astype(int)
                    hard_labels[i, curr_labels] = 1
            else:
                hard_labels = y_tilt.detach().cpu().numpy()
                masked_alphas = alphas
        
        return hard_labels, total_loss, masked_alphas

 
    def beam_search(v_prob, v_feat, device, max_label_len=10,
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
        
    
    
class CNN_RNN(nn.Module):
    """
        Wrapper class to combine CNN with visual attention RNN
    """
    def __init__(self, args, device):
        super(CNN_RNN, self).__init__()
        self.args = args
        self.device = device
        self.encoder = EncoderCNN(args)
        self.decoder = AttnDecoderRNN(ATTN_DIM, DECODER_DIM, ENCODER_DIM, dropout=args.dropout)
    
    def forward(self, x, labels=None, loss_fn=None, is_eval=False):
        v_prob, v_feat = self.encoder(x)

        hard_labels, total_loss, alphas = self.decoder(v_prob, 
                                                       v_feat, 
                                                       self.device, 
                                                       max_label_len=MAX_LABEL_LEN, 
                                                       prob_path_thresh=self.args.prob_path_thresh, 
                                                       loss_fn=loss_fn,
                                                       labels=labels, 
                                                       is_eval=is_eval, 
                                                       beam_search=self.args.beam_search,
                                                       beam_size=self.args.beam_size)

        return total_loss, hard_labels, alphas
        
        
        
        
    
    
    
