import argparse
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets

from model import UNet
from dataset import *
from utils import *

# Parser 생성
parser = argparse.ArgumentParser(description='Train the Unet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser에 사용할 argument add
parser.add_argument('--lr',default=1e-3,type=float,dest='lr')
parser.add_argument('--batch_size',default=4,type=float,dest='batch_size')
parser.add_argument('--num_epoch',default=100,type=float,dest='num_epoch')

parser.add_argument('--data_dir',default="./datasets",type=str,dest='data_dir')
parser.add_argument('--ckpt_dir',default='./checkpoint',type=str,dest='ckpt_dir')
parser.add_argument('--log_dir',default='./log',type=str,dest='log_dir')
parser.add_argument('--result_dir',default='./result',type=str,dest='result_dir')

parser.add_argument('--mode',default='train',type=str,dest='mode')
parser.add_argument('train_continue',default='off',type=str, dest='train_continue')

# parser를 args에 할당
args = parser.parse_args()

## 하이퍼 파라미터 설정
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

print('lr : %.4e'%lr)
print('batch : %d'% batch_size)

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir,'png'))
    os.makedirs(os.path.join(result_dir,'numpy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 네트워크 선언
if mode == 'train':
    # transform 적용해서 데이터 셋 불러오기
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'),transform=transform)

    # 불러온 데이터셋, 배치 size줘서 DataLoader 해주기
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle=True)

    # val set도 동일하게 진행
    dataset_val = Dataset(data_dir=os.path.join(data_dir,'val'),transform = transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size , shuffle=True)

    # 기타 variables 설정
    num_train = len(dataset_train)
    num_val = len(dataset_val)

    num_train_for_epoch = np.ceil(num_train/batch_size) # np.ceil : 소수점 반올림
    num_val_for_epoch = np.ceil(num_val/batch_size)
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)

    # 불러온 테스트 셋, 배치 size줘서 DataLoader 해주기
    # shuffle 안함(test 이므로)
    loader_test = DataLoader(dataset_test, batch_size=batch_size)

    # 네트워크 불러오기
    net = UNet().to(device)  # device : cpu or gpu

    # loss, optimizer define
    fn_loss = nn.BCEWithLogitsLoss().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    # 기타 variables 설정
    num_test = len(dataset_test)
    num_test_for_epoch = np.ceil(num_test / batch_size)  # np.ceil : 소수점 반올림

# 네트워크 불러오기
net = UNet().to(device) # device : cpu or gpu

# loss 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 정의
optim = torch.optim.Adam(net.parameters(), lr = lr )

# 기타 function 설정
fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0,2,3,1) # device 위에 올라간 텐서를 detach 한 뒤 numpy로 변환
fn_denorm = lambda x, mean, std : (x * std) + mean
fn_classifier = lambda x :  1.0 * (x > 0.5)  # threshold 0.5 기준으로 indicator function으로 classifier 구현

# Tensorbord
writer_train = SummaryWriter(log_dir=os.path.join(log_dir,'train'))
writer_val = SummaryWriter(log_dir = os.path.join(log_dir,'val'))


## 네트워크 학습 및 평가
start_epoch = 0
## Train mode
if mode == 'train':
    if train_continue == 'on': # 학습을 계속 할거면 저장된 네트워크 불러오기
        net, optim, start_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)  # 저장된 네트워크 불러오기

    for epoch in range(start_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):  # 1은 뭐니 > index start point
            # forward
            label = data['label'].to(device)  # 데이터 device로 올리기
            inputs = data['input'].to(device)
            output = net(inputs)

            # backward
            optim.zero_grad()  # gradient 초기화
            loss = fn_loss(output, label)  # output과 label 사이의 loss 계산
            loss.backward()  # gradient backpropagation
            optim.step()  # backpropa 된 gradient를 이용해서 각 layer의 parameters update

            # save loss
            loss_arr += [loss.item()]

            # tensorbord에 결과값들 저정하기
            label = fn_tonumpy(label)
            inputs = fn_tonumpy(fn_denorm(inputs, 0.5, 0.5))
            output = fn_tonumpy(fn_classifier(output))

            writer_train.add_image('label', label, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', inputs, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    # validation
    with torch.no_grad():  # validation 이기 때문에 backpropa 진행 x, 학습된 네트워크가 정답과 얼마나 가까운지 loss만 계산
        net.eval()  # 네트워크를 evaluation 용으로 선언
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            output = net(inputs)

            # loss
            loss = fn_loss(output, label)
            loss_arr += [loss.item()]
            print('valid : epoch %04d / %04d | Batch %04d \ %04d | Loss %04d' % (
            epoch, num_epoch, batch, num_val_for_epoch, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_classifier(output))

            writer_val.add_image('label', label, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('input', inputs, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        # epoch이 끝날때 마다 네트워크 저장
        if epoch % 5 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()
else:
    net, optim, start_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():  # test 이기 때문에 backpropa 진행 x, 학습된 네트워크가 정답과 얼마나 가까운지 loss만 계산
        net.eval()  # 네트워크를 evaluation 용으로 선언
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            output = net(inputs)

            # loss
            loss = fn_loss(output, label)
            loss_arr += [loss.item()]
            print('Test : Batch %04d \ %04d | Loss %.4f' % (batch, num_test_for_epoch, np.mean(loss_arr)))

            # output을 numpy와 png 파일로 저장
            label = fn_tonumpy(label)
            inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_classifier(output))

            for j in range(label.shape[0]):
                id = num_test_for_epoch * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'inputs_%04d.png' % id), inputs[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.np' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'inputs_%04d.np' % id), inputs[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.np' % id), output[j].squeeze())

    print('Average Test : Batch %04d \ %04d | Loss %.4f' % (batch, num_test_for_epoch, np.mean(loss_arr)))