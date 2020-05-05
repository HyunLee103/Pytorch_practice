import os
import numpy as np

import torch
import torch.nn as nn

from layer import *

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels,nker=128,norm="bnorm"):
        super(Generator,self).__init__()

        self.dec1 = DECBR2d(1 * in_channels, 8 * nker, kernel_size=4,stride=1,
                            padding=0,norm=norm,relu=0.0,bias=False)
        self.dec2 = DECBR2d(8 * nker,4 * nker, kernel_size=4,stride=2,
                            padding=1, norm=norm,relu=0.0, bias=False)
        self.dec3 = DECBR2d(4 * nker,2 * nker, kernel_size=4,stride=2,
                            padding=1, norm=norm,relu=0.0, bias=False)
        self.dec4 = DECBR2d(2 * nker,1 * nker, kernel_size=4,stride=2,
                            padding=1, norm=norm,relu=0.0, bias=False)
        self.dec5 = DECBR2d(1 * nker,out_channels, kernel_size=4,stride=2,
                            padding=1, norm=None,relu=None, bias=False)

    def forward(self,x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)

        x = torch.tanh(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels,out_channels,nker=64,norm='bnorm'):
        super(Discriminator,self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)   # relu=0.2 > LekyLely
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc3 = CBR2d(2*nker, 4 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc5 = CBR2d(8 * nker, out_channels, kernel_size=4,stride=2,
                          padding=1,norm=None,relu=None,bias=False)

    def forward(self,x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        x = torch.sigmoid(x)

        return x


