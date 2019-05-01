import torch
import torch.nn as nn
import torchvision as vision
from constants.constants import NUM_CLASSES

class ResNet50(nn.Module):
    def __init__(self, args):
        super(ResNet50, self).__init__()
        
        if args.use_pretrained:
            self.resnet50 = vision.models.resnet50(pretrained=True)
            #self.resnet50 = vision.models.densenet121(pretrained=True)
        else:
            self.resnet50 = vision.models.resnet50(pretrained=False)
            self.resnet50.load_state_dict(torch.load(args.load_path))
        
        if args.feature_extracting:
            self.set_parameter_requires_grad(self.resnet50, feature_extracting=True, nlayers_to_freeze=None)
            num_ftrs = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_ftrs, NUM_CLASSES)
            #self.resnet50.fc = MultiLayerPerceptron(num_ftrs)
            
            # TEMP: densenet121
            #num_ftrs = self.resnet50.classifier.in_features
            #self.resnet50.classifier = MultiLayerPerceptron(num_ftrs)
            
        else:
            print('Fine-tune ResNet50 with {} layers freezed...'.format(args.nlayers_to_freeze))
            self.set_parameter_requires_grad(self.resnet50, 
                                             feature_extracting=False,
                                             nlayers_to_freeze=args.nlayers_to_freeze)
            num_ftrs = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        
    def forward(self, x):
        return self.resnet50(x)
    
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
