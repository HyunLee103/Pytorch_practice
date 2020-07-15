import os
import numpy as np

import torch
import torch.nn as nn

from layer import *

# Convolutions in the encoder, and in the discriminator, downsample
# by a factor of 2, whereas in the decoder they upsample by a factor of 2.

class Pix2Pix_generator(nn.Module):
    def __init__(self, in_channels,out_channels,nker=64,norm='bnorm'):
        super(Pix2Pix, self).__init__()

        # encoder
        # Leaky relu 사용, 첫번째 encoder는 batchnorm X
        self.enc1 = CBR2d(in_channels,1*nker,kernel_size=4, padding=1,stride=2,
        norm = None,relu=0.2)
        self.enc2 = CBR2d(1*nker,2*nker,kernel_size=4, padding=1,stride=2,
        norm = norm ,relu=0.2)
        self.enc3 = CBR2d(2*nker,4*nker,kernel_size=4, padding=1,stride=2,
        norm = norm,relu=0.2)
        self.enc4 = CBR2d(4*nker,8*nker,kernel_size=4, padding=1,stride=2,
        norm = norm,relu=0.2)
        self.enc5 = CBR2d(8*nker,8*nker,kernel_size=4, padding=1,stride=2,
        norm = norm,relu=0.2)
        self.enc6 = CBR2d(8*nker,8*nker,kernel_size=4, padding=1,stride=2,
        norm = norm,relu=0.2)
        self.enc7 = CBR2d(8*nker,8*nker,kernel_size=4, padding=1,stride=2,
        norm = norm,relu=0.2)
        self.enc8 = CBR2d(8*nker,8*nker,kernel_size=4, padding=1,stride=2,
        norm = norm,relu=0.2)

        # decoder, skip-connection 고려해서 input channel modeling
        self.dec1 = DECBR2d(8*nker, 8*nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)
        self.drop1 = nn.Dropout2d(0.5)
        
        self.dec2 = DECBR2d(2 * 8 * nker, 8*nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)
        self.drop2 = nn.Dropout2d(0.5)

        self.dec3 = DECBR2d(2*8*nker, 8*nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)
        self.drop3 = nn.Dropout2d(0.5)

        self.dec4 = DECBR2d(2 * 8*nker, 8*nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)
        
        self.dec5 = DECBR2d(2*8*nker, 4*nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)
        
        self.dec6 = DECBR2d(2*4*nker, 2* nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)
       
        self.dec7 = DECBR2d(2*2*nker, 1*nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)

        self.dec8 = DECBR2d(2*1*nker, out_channels, kernel_size=4, padding=1,
        norm = None, relu=None, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)

        dec1 = self.dec1(enc8)
        drop1 = self.drop1(dec1)

        cat2 = torch.cat((drop1,enc7),dim=1)
        dec2 = self.dec2(cat2)
        drop2 = self.drop2(dec2)

        cat3 = torch.cat((drop2,enc6),dim=1)
        dec3 = self.dec3(cat3)
        drop3 = self.drop3(dec3)

        cat4 = torch.cat((drop3,enc5),dim=1)
        dec4 = self.dec4(cat4)

        cat5 = torch.cat((dec4,enc4),dim=1)
        dec5 = self.dec5(cat5)

        cat6 = torch.cat((dec5,enc3),dim=1)
        dec6 = self.dec6(cat6)
        
        cat7 = torch.cat((dec6,enc2),dim=1)
        dec7 = self.dec7(cat7)
        
        cat8 = torch.cat((dec7,enc1),dim=1)
        dec8 = self.dec8(cat8)

        x = torch.tanh(dec8)

        return x


class Pix2Pix_Discriminator(nn.Module):
    def __init__(self, in_channels,out_channels,nker=64,norm='bnorm'):
        super(Pix2Pix_Discriminator,self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4,stride=2,
                          padding=1,norm=None,relu=0.2,bias=False)   # 첫번째 D layer에는 batch 적용 X 
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


