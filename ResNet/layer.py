import torch
import torch.nn as nn


# model에서 반복되는 layer를 묶어서 하나의 class로 선언
# model 선언 할때 nn.Module을 상속 받기 때문에 계층을 맞춰주기 위해 CBR2d class도 nn.Module 상속 받음
class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, norm ="bnorm" ,relu = 0.0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]
        if not norm is None:
            if norm == 'bnorm':
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == 'inorm':
                layers += [nn.InstanceNorm2d(num_features=out_channels)]
        if not relu is None:
            layers += [nn.ReLU() if relu == 0.0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self,x):
        return self.cbr(x)


class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, norm ="bnorm" ,relu = 0.0):
        super().__init__()

        layers = []
        layers += [CBR2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,
                         padding=padding, bias=bias,norm=norm, relu=relu)]
        layers += [CBR2d(out_channels,out_channels,kernel_size=kernel_size,stride=stride,
                         padding=padding,bias=bias,norm=norm,relu = None)]

        self.resblk = nn.Sequential(*layers)

    def forward(self,x):
        return x + self.resblk(x)

class PixelUnShuffle(nn.Module):
    def __init__(self,ry=2,rx =2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B,C,H,W] = list(x.shape)

        x = x.reshape(B,C,H // ry,ry,W//rx,rx)
        x = permute(0,1,3,5,2,4)
        x = x.reshape(B,C*ry*rx,H//ry,W//rx)

        return x

class PixelShuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx,H,W)
        x = permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // (ry * rx), H*ry, W*rx)

        return x


















