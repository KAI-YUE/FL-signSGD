import math
import numpy as np

# Pytorch libraries
import torch.nn as nn
import torch.nn.functional as F

class NaiveMLP(nn.Module):
    def __init__(self, dim_in,  dim_out, dim_hidden=50):
        super(NaiveMLP, self).__init__()
        self.predictor = nn.Sequential(
                            nn.Linear(dim_in, dim_hidden),
                            nn.ReLU(),
                            nn.Linear(dim_hidden, dim_out)
                         )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.predictor(x)


class NaiveCNN(nn.Module):
    def __init__(self, channels=1, dim_out=10, **kwargs):
        super(NaiveCNN, self).__init__()
        self.channels = channels
        if "dim_in" in kwargs:
            self.input_size = int(np.sqrt(kwargs["dim_in"]/channels))
        else:
            self.input_size = 28
        
        kernel_size = 3
        self.fc_input_size = (((((self.input_size - kernel_size)/1 + 1) - kernel_size)/1 + 1) - kernel_size)/2 + 1
        self.fc_input_size = int(self.fc_input_size)**2 * 20

        self.predictor = nn.Sequential(
                    nn.Conv2d(channels, 10, kernel_size=kernel_size),
                    nn.ReLU(),
                    nn.Conv2d(10, 20, kernel_size=kernel_size),
                    nn.MaxPool2d(kernel_size=kernel_size, stride=2),
                    nn.ReLU(),
                    )
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], self.channels, self.input_size, self.input_size)
        x = self.predictor(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FullPrecision_BN(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(FullPrecision_BN, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2048, 512)
        self.bn3 = nn.BatchNorm1d(512, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(512, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False


def init_weights(module, init_type='kaiming', gain=0.01):
    '''
    initialize network's weights
    init_type: normal | uniform | kaiming  
    '''
    classname = module.__class__.__name__
    if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, gain)
        elif init_type == "uniform":
            nn.init.uniform_(module.weight.data)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')

        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find('BatchNorm') != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find("GroupNorm") != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)
