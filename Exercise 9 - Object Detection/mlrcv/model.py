from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from mlrcv.utils import *

class CenterNet(nn.Module):
    def __init__(self, n_classes=1, model_name="resnet18"):
        super(CenterNet, self).__init__()
        # create backbone.
        basemodel = torchvision.models.resnet18(pretrained=False) # turn this on for training
        basemodel = nn.Sequential(*list(basemodel.children())[:-2])
        # set basemodel
        self.base_model = basemodel

        self.up1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # output classification
        self.outc = nn.Conv2d(256, n_classes, 1)
        # output residue
        self.outr = nn.Conv2d(256, 2, 1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.base_model(x)

        # Add positional info        
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)

        outc = self.outc(x)
        outr = self.outr(x)
        
        return outc, outr