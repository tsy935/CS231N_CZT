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

    def forward(self, v_prob, v_feat, device, max_label_len=10, prob_path_thresh=1e-6, loss_fn=None, labels=None, is_eval=False):
        """
        Forward function for training only.
        Args:
            v_prob: predicted probability from CNN, (batch_size, NUM_CLASSES)
            v_feat: visual map features extracted from CNN
            device: device
            max_label_len: maximum label length in train split
            prob_path_thresh: prediction path probability threshold to terminate test prediction
            labels: one-hot encoded ground truth labels, (batch_size, ncrops, NUM_CLASSES)
        """
        batch_size = v_feat.size(0)
        encoder_dim = v_feat.size(-1)
        
        v_feat = v_feat.view(batch_size, -1, encoder_dim) # (batch_size, num_pixels, encoder_dim)
        num_pixels = v_feat.size(1)
                
        # Initialize LSTM state
        h, c = self.init_hidden_state(v_feat)  # (batch_size, decoder_dim)
        
        # Initialize v_pred
        v_pred = v_prob # (batch_size, NUM_CLASSES)
        
        # Initialize candidate pool
        #C = np.tile(np.arange(0,NUM_CLASSES), (batch_size, 1)) # (batch_size, NUM_CLASSES)
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
            prob_path = np.ones(batch_size, max_label_len)
            hard_labels = np.zeros(batch_size, NUM_CLASSES)
            for t in range(max_label_len):
                prob_path[:,t] = np.prod(soft_probs[:,:t+1], axis=1)
                
            masks = prob_path >= prob_path_thresh
            
            masked_alphas = alphas * masks
            for i in range(batch_size):
                curr_labels = y_tilt_num[i, masks[i,:]]
                hard_labels[i, int(curr_labels)] = 1
        else:
            hard_labels = y_tilt.detach().cpu().numpy()
            masked_alphas = alphas
        
        return hard_labels, total_loss, masked_alphas

 
    
#    def beam_search(self, v_prob, v_feat, device, beam_size=3):
#        # TODO: finish beam search!!!
#        """
#        Function to perform beam search.
#        Args:
#            v_prob: predicted probability from CNN for one single example (1, NUM_CLASSES)
#            v_feat: visual map features extracted from CNN for one single example
#            device: device
#            beam_size: size of beam
#        """
##        batch_size = v_feat.size(0)
#        encoder_dim = v_feat.size(-1)
#        enc_image_size = v_feat.size(1)
#        
#        v_feat = v_feat.view(-1, encoder_dim) # (num_pixels, encoder_dim)
#        num_pixels = v_feat.size(0)
#        
#        k = beam_size
#        
#        # We'll treat the problem as having a batch size of k
#        v_feat = v_feat.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
#    
#        # Tensor to store top k previous words at each step; now they're just label 0's
#        k_prev_labels = torch.LongTensor([[0]] * k).to(device)  # (k, 1)
#    
#        # Tensor to store top k sequences; now they're just label 0's
#        seqs = k_prev_labels  # (k, 1)
#    
#        # Tensor to store top k sequences' scores; now they're just 0
#        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
#    
#        # Tensor to store top k sequences' alphas; now they're just 1s
#        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)
#    
#        # Lists to store completed sequences, their alphas and scores
#        complete_seqs = list()
#        complete_seqs_alpha = list()
#        complete_seqs_scores = list()
#    
#        # Initialization
#        step = 1
#        h, c = self.init_hidden_state(v_feat)
#        v_pred = v_prob.expand(k, NUM_CLASSES) # (k, NUM_CLASSES)
#        C = np.tile(np.arange(0,NUM_CLASSES), (k, 1)) # (k, NUM_CLASSES)
#        
#        y_tilt = torch.zeros_like(v_pred).to(device) # (k, NUM_CLASSES)
#    
#        # s is a number less than or equal to k, because sequences are removed from this 
#        # process once they reach stopping conditions
#        probs = torch.ones_like(v_pred)
#        while True:    
#            z, alpha = self.attn(v_feat, h)    
##            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)   
#            gate = self.sigmoid(self.f_beta(h))
#            z = gate * z
#            
#            h, c = self.decode_step(torch.cat([v_pred, z, y_tilt], dim=1), (h, c))
#            fc1_out = self.fc1(h)
#            logit = self.fc2(self.dropout(fc1_out)) # (s, NUM_CLASSES)
#            p = self.sigmoid(logit)
##            probs[:,t,:] = p
##            alphas[:,t,:] = alpha
#            v_pred = p
#            
#            # Only choose from the candidate pool
#            p_C = p[:,C].numpy()
#            curr_idx = np.argmax(p_C, dim=1)
#            l = C[curr_idx] # current time step hard prediction
#            y_tilt[:,int(l)] = 1 # add current time step hard prediction
#            C = np.delete(C, curr_idx) # remove newly predicted label from candidate pool
#            
#            # Add 
#            scores = top_k_scores.expand_as(p) + p  # (s, NUM_CLASSES)
#   
#            # For the first step, all k points will have the same scores (since same k previous words, h, c)
#            if step == 1:
#                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
#            else:
#                # Unroll and find top scores, and their unrolled indices
#                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
#    
#            # Convert unrolled indices to actual indices of scores
#            prev_word_inds = top_k_words / NUM_CLASSES  # (s)
#            next_word_inds = top_k_words % NUM_CLASSES  # (s)
#    
#            # Add new words to sequences, alphas
#            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
#            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
#                                   dim=1)  # (s, step+1, enc_image_size, enc_image_size)
#    
#            # Which sequences hit one of the two termination conditions?
#            complete_inds = scores < args.beam_search_prob_thresh
#            
#            
#            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
#                               next_word != word_map['<end>']]
#            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
#    
#            # Set aside complete sequences
#            if len(complete_inds) > 0:
#                complete_seqs.extend(seqs[complete_inds].tolist())
#                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
#                complete_seqs_scores.extend(top_k_scores[complete_inds])
#            k -= len(complete_inds)  # reduce beam length accordingly
#    
#            # Proceed with incomplete sequences
#            if k == 0:
#                break
#            seqs = seqs[incomplete_inds]
#            seqs_alpha = seqs_alpha[incomplete_inds]
#            h = h[prev_word_inds[incomplete_inds]]
#            c = c[prev_word_inds[incomplete_inds]]
#            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
#            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
#            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
#    
#            # Break if things have been going on too long
#            if step > 50:
#                break
#            step += 1
#    
#        i = complete_seqs_scores.index(max(complete_seqs_scores))
#        seq = complete_seqs[i]
#        alphas = complete_seqs_alpha[i]
#    
#        return seq, alphas
    
    
class CNN_RNN(nn.Module):
    """
        Wrapper class to combine CNN with visual attention RNN
    """
    def __init__(self, args, device, is_eval=False):
        super(CNN_RNN, self).__init__()
        self.args = args
        self.device = device
        self.is_eval = is_eval
        self.encoder = EncoderCNN(args)
        self.decoder = AttnDecoderRNN(ATTN_DIM, DECODER_DIM, ENCODER_DIM, dropout=args.dropout)
    
    def forward(self, x, labels=None, loss_fn=None):
        v_prob, v_feat = self.encoder(x)

        hard_labels, total_loss, alphas = self.decoder(v_prob, v_feat, self.device, max_label_len=MAX_LABEL_LEN, 
                                           prob_path_thresh=self.args.prob_path_thresh, loss_fn=loss_fn,
                                           labels=labels, is_eval=self.is_eval)
        if labels is None:
            return hard_labels, alphas
        else:
            return total_loss, hard_labels, alphas
        
        
        
        
    
    
    