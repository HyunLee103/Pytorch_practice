import os
import numpy as np
import torch
import torch.nn as nn

from layer import * # layer에 CBR2d class import


## 네트워크 구축
class UNet(nn.Module):
    def __init__(self,in_channels ,out_channels,nch,nker,norm = 'bnorm',learning_type ='plain'):
        super(UNet, self).__init__()
        self.learning_type = learning_type

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=in_channels, out_channels=1*nker, norm=norm)
        self.enc1_2 = CBR2d(in_channels=1*nker, out_channels=1*nker,norm=norm)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=1*nker, out_channels=2*nker,norm=norm)
        self.enc2_2 = CBR2d(in_channels=2*nker, out_channels=2*nker,norm=norm)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=2*nker, out_channels=4*nker,norm=norm)
        self.enc3_2 = CBR2d(in_channels=4*nker, out_channels=4*nker,norm=norm)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=4*nker, out_channels=8*nker,norm=norm)
        self.enc4_2 = CBR2d(in_channels=8*nker, out_channels=8*nker,norm=norm)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=8*nker, out_channels=16*nker,norm=norm)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=16*nker, out_channels=8*nker,norm=norm)

        self.unpool4 = nn.ConvTranspose2d(in_channels=8*nker, out_channels=8*nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 8*nker, out_channels=8*nker,norm=norm)
        self.dec4_1 = CBR2d(in_channels=8*nker, out_channels=4*nker,norm=norm)

        self.unpool3 = nn.ConvTranspose2d(in_channels=4*nker, out_channels=4*nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2* 4*nker, out_channels=4*nker,norm=norm)
        self.dec3_1 = CBR2d(in_channels=4*nker, out_channels=2*nker,norm=norm)

        self.unpool2 = nn.ConvTranspose2d(in_channels=2*nker, out_channels=2*nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 2*nker, out_channels=2*nker,norm=norm)
        self.dec2_1 = CBR2d(in_channels=2*nker, out_channels=1*nker,norm=norm)

        self.unpool1 = nn.ConvTranspose2d(in_channels=1*nker, out_channels=1*nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 1*nker, out_channels=1*nker,norm=norm)
        self.dec1_1 = CBR2d(in_channels=1*nker, out_channels=1*nker,norm=norm)

        self.fc = nn.Conv2d(in_channels=1*nker, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        if self.learning_type == 'plain':
            out = self.fc(dec1_1)
        elif self.learning_type == 'residual':
            out = self.fc(dec1_1) + x   # output과 네트워크 인풋을 더해 최종 아웃풋으로 낸다(residual learning)

        return out


class SRResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, learning_type = 'plain',
                 norm = 'bnorm',nblk = 16):
        super(SRResNet,self).__init__()

        self.learning_type = learning_type
        self.enc = CBR2d(in_channels,nker,kernel_size=9,stride=1,padding=4,
                         bias=True,norm=None,relu = 0.0)

        res = []

        for i in range(nblk):
            res += [Resblock(nker,nker,kernel_size=3,stride=1,padding=1,bias=True,
                             norm=norm,relu=0.0)]
        self.res = nn.Sequential(*res)

        self.dec = CBR2d(nker,nker,kernel_size=3,stride=1,padding=1,norm=norm,
                         bias=True,relu =None)

        ps1 = []
        ps1 += [nn.Conv2d(in_channels=nker,out_channels=4*nker,kernel_size=3,
                          stride=1,padding=1)]
        ps1 += [PixelShuffle(ry=2,rx=2)]
        ps1 += [nn.ReLU()]
        self.ps1 = nn.Sequential(*ps1)

        ps2 = []
        ps2 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3,
                          stride=1, padding=1)]
        ps2 += [PixelShuffle(ry=2, rx=2)]
        ps2 += [nn.ReLU()]
        self.ps2 = nn.Sequential(*ps2)

        self.fc = nn.Conv2d(in_channels = nker, out_channels=out_channels,
                              kernel_size=9,stride = 1, padding=4)

    def forward(self,x):
        out = self.enc(x)
        x0 = x

        out = self.res(out)
        out = self.dec(out)
        out = x0 + out  # skip connection

        out = self.ps1(out)
        out = self.ps2(out)

        out = self.fc(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channels,out_channels,nker=64,learning_type='plain',
                 norm = ' bnorm',nblk = 16):
        super(ResNet,self).__init__()
        self.learning_type = learning_type

        self.enc = CBR2d(in_channels=in_channels,out_channels=nker,
                         kernel_size=3, stride=1,
                         padding=1, bias=True, norm=None, relu=0.0)
        res = []
        for i in range(nblk):
            res += [Resblock(nker,nker,kernel_size=3,stride=1,
                             padding=1,bias=True,norm=norm,relu=0.0)]
        self.res = nn.Sequential(*res)

        self.dec = CBR2d(nker,nker,kernel_size=3,stride=1,
                         padding=1,bias=True,norm=norm,relu=0.0)

        self.fc = nn.Conv2d(in_channels=nker,out_channels=out_channels,
                            kernel_size=1,stride=1,padding=0,bias=True)

    def forward(self,x):
        x0 = x

        out = self.enc(x)
        out = self.res(out)
        out = self.dec(out)

        if self.learning_type == "plain":
            out = self.fc(out)
        elif self.learning_type == 'residual':
            out = self.fc(out) + x0

        return out




