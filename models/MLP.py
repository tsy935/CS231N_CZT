import torch
import torch.nn as nn
import torchvision as vision
from constants.constants import NUM_CLASSES

class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_ftrs):
        super(MultiLayerPerceptron, self).__init__()
        self.linear1 = nn.Linear(num_ftrs, num_ftrs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(num_ftrs, NUM_CLASSES)
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)