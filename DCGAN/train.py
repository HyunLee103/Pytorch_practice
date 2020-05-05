## 라이브러리 추가하기
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *


import matplotlib.pyplot as plt

from torchvision import transforms

## Parser 생성하기
parser = argparse.ArgumentParser(description="DCGAN modeling",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets/BSR/BSDS500/data/images", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--task", default="DCGAN", choices=['DCGAN'], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['bilinear', 4.0, 0], dest='opts')
parser.add_argument("--ny", default=64, type=int, dest="ny")
parser.add_argument("--nx", default=64, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=128, type=int, dest="nker")

parser.add_argument("--network", default="DCGAN", choices=["unet", "hourglass", "resnet", "srresnet",'DCGAN'], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()

## 트레이닝 파라메터 설정하기
mode = args.mode
train_continue = args.train_continue

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network
learning_type = args.learning_type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("mode: %s" % mode)

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)

print("task: %s" % task)
print("opts: %s" % opts)

print("network: %s" % network)
print("learning type: %s" % learning_type)

print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)

print("device: %s" % device)

## 디렉토리 생성하기
result_dir_train = os.path.join(result_dir, 'train')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir_train):
    os.makedirs(os.path.join(result_dir_train, 'png'))
if not os.path.exists(result_dir_test):
    os.makedirs(os.path.join(result_dir_test, 'png'))


## 데이터 로더, generative model은 test, val dataset 없다!. 랜덤 벡터로 부터 생성된 이미지만 확인하면 되므로
if mode == 'train':
    transform_train = transforms.Compose([Resize(shape=(ny,nx,nch)), Normalization(mean=0.5,std=0.5)]) # generator output인 tanh(-1~1) 값과
                                                                                                       # D input scale을 맞춰주기 위해 normalize
    dataset_train = Dataset(data_dir=os.path.join(data_dir), transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_batch_train = np.ceil(num_data_train / batch_size)




## 네트워크 생성하기
if network == "DCGAN":
    netG = Generator(in_channels=100,out_channels=nch,nker=nker).to(device)
    netD = Discriminator(in_channels=nch,out_channels=1,nker=nker).to(device)

    # 네트워크 가중치를 정규분포, std 0.02로 초기화
    init_weights(netG,init_type='normal',init_gain=0.02)
    init_weights(netD,init_type='normal',init_gain=0.02)

## 손실함수 정의하기
fn_loss = nn.BCELoss().to(device)

## Optimizer 설정하기
## 하나의 network에 하나의 optimizer가 필요, GAN은 네트워크가 두개 이므로 optimizer도 두개 필요
optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5,0.999)) # 논문에 맞게 Adam beta term을 조정 해줘야함
optimD = torch.optim.Adam(netD.parameters(), lr=lr,betas=(0.5,0.999))



## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)
cmap = None

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0

# TRAIN MODE
if mode == 'train':
    if train_continue == "on": # 이어서 학습할때 on, 처음 학습 시킬때는 off
        netG,netD, optimG,optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG,netD=netD, optimG=optimG,optimD=optimD)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        netG.train() # 사용할 네트워크를 train 모드로 설정
        netD.train() # 사용할 네트워크를 train 모드로 설정

        loss_G_train= []
        loss_D_real_train = []
        loss_D_fake_train = [] # D(G(z))

        for batch, data in enumerate(loader_train, 1): # loader_train에는 real-image data가 들어있다.
            # forward pass
            label = data['label'].to(device) # label : 실제이미지 data, device로 올리기 > D의 real 파트 인풋
            input = torch.randn(label.shape[0],100,1,1).to(device)  # generator 인풋은 유니폼 분포에서 랜덤 샘플된 100차원 벡터
                                                                    # label.shape[0] : batch_size, 100 : ch , 1 : H, 1 : W
            output = netG(input) # 위에서 생성한 noise input을 generator에 넣어 하나의 이미지 생성

            # backward pass
            # backprop도 두 네트워크 각각 해줘야 한다.

            # Discriminator backprop
            set_requires_grad(netD,True) # netD(Discriminator)의 모든 파라미터 연산을 추적해 gradient를 계산한다.
            optimD.zero_grad() # gradient 초기화

            pred_real = netD(label)  # 진짜 이미지를 Discriminator에 통과시킨다. 이 결과가 True(1)를 가지게 D를 학습
            pred_fake = netD(output.detach()) # output = netG(input), 즉 generator로 생성된 가짜 이미지를 D에 넣는다.
                                              # 이 결과를 False(0)로 구분하게 D를 학습
                                              # detach()는 G(z)-output을 그대로 넣으면 Generator까지 gradient가 흘러가기때문에
                                              # 이를 분리해주기 위해 사용한다.
            # D의 loss는 가짜이미지와 진짜이미지를 잘 구분하게 형성되어야 한다.
            # 즉 G로 만들어진 가짜이미지는 False(0)으로 진짜 이미지는 True(1)로 타겟을 주고 loss를 구한다
            loss_D_real = fn_loss(pred_real,torch.ones_like(pred_real))
            loss_D_fake = fn_loss(pred_fake,torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_fake + loss_D_real)
            loss_D.backward() # gradient backprop
            optimD.step() # gradient update

            # Generator backprop
            set_requires_grad(netD,False) # generator를 학습할땐, discriminator는 고정한다. 따라서 required_grad = False
            optimG.zero_grad() # gradient 초기화

            pred_fake = netD(output)

            loss_G = fn_loss(pred_fake,torch.ones_like(pred_fake))  # generator는 생성한 가짜 이미지가 진짜(True,1)로 분류되게 학습해야한다.

            loss_G.backward() # gradient backprop
            optimG.step() # gradient update

            # 손실함수 계산
            loss_G_train += [loss_G.item()]
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | G_LOSS %.4f | D_fake_LOSS %.4f | D_real_LOSS %.4f " %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_G_train), np.mean(loss_D_fake_train),np.mean(loss_D_real_train)))

            if batch % 10 == 0:
              # Tensorboard 저장하기
              output = fn_tonumpy(fn_denorm(output,mean=0.5,std=0.5)).squeeze() # generator로 생성된 output은 마지막 layer에 tanh를 거치며 normalize된다.
                                                                                # 따라서 denorm을 통해 복원
              id = num_batch_train * (epoch - 1) + batch

              plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)
              writer_train.add_image('output', output, id, dataformats='NHWC')

        writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)
        writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)
        writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)

    # generative model은 unsupervised-learning이기 때문에, val 부분이 없다.

        if epoch % 2 == 0: # epoch 2번마다 네트워크 checkpoint 저장
            save(ckpt_dir=ckpt_dir, netG=netG, netD = netD, optimG=optimG, optimD=optimD, epoch=epoch)

    writer_train.close() # tensorboard 닫기


# TEST MODE
else:
    netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD = netD, optimG=optimG, optimD = optimD)

    with torch.no_grad():
        netG.eval()  # generative model은 test에서 Generator net만 사용한다. 왜냐면! generator로 얼마나 이미지가 잘 생성되었는지 보면 되므로
                     # 따라서 netG만 eval 모드로 해주면 된다. 따로 test loss는 없다

        # input은 유니폼 분포의 랜덤 샘플 벡터
        input = torch.randn(batch_size,100,1,1).to(device)

        output = netG(input)
        output = fn_tonumpy(fn_denorm(output, mean=0.5,std=0.5)) # generator의 아웃풋이 tanh로 normalize 되었으므로 denorm

        for j in range(output.shape[0]):
            id =  j

            output_ = output[j]
            plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)

