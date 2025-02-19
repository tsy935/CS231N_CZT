#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 00:53:02 2019

@author: rugezhao
"""

import torch
import torch.nn as nn
import torchvision as vision
from constants.constants import NUM_CLASSES
from models.MLP import MultiLayerPerceptron
#TODO: writes a link to HOG data: aim is the feed HOG into the last FC layer as part of input

class ResNet50_HOGFC(nn.Module):
    def __init__(self, args):
        super(ResNet50_HOGFC, self).__init__()
        
        
        if args.use_pretrained and (args.resnet_path is None):
            self.resnet50 = vision.models.resnet50(pretrained=True) # load ImageNet pretrained weights
        elif args.use_pretrained and (args.resnet_path is not None):
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
            
        # modified input size of the last fc layer. added 27*27*36 which is the vector size using HOG for each image
            
        # Remove linear layer
        modules = list(self.resnet50.children())[:-1]
        self.conv_features = nn.Sequential(*modules) # all layers until last pool layer (inclusive)
            
        self.hogfc = nn.Linear(num_ftrs+10368, NUM_CLASSES)
        
        # one layer fc works, we can run MLP
        # self.hogmlp = MultiLayerPerceptron(num_ftrs+10368)
            
        
    def forward(self, x, hog_features):
        # intermediate result before fc
        resnet_features = self.conv_features(x) 
        # batch size
        B = resnet_features.shape[0]
        # flatten resnet_features
        resnet_features = resnet_features.reshape(B,-1)
        # concatenate with hog feature
         
        # convert to (batch_size*6, hog_vector_size)
        hog_features = hog_features.reshape(B,-1)
        features_concat = torch.cat((resnet_features.float(), hog_features.float()),dim=1)
        # fc
        scores = self.hogfc(features_concat)
        # mlp        
        # scores = self.hogmlp(features_concat)
        return scores
    
    def set_parameter_requires_grad(self, model, feature_extracting=False, nlayers_to_freeze=None):
        # freeze nlayers_to_freeze layers
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


