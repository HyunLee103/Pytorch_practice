## 라이브러리 추가하기

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

def train(args):
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

    wgt = args.wgt

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
    
    ## 데이터 로더, DCGAN은 test, val dataset 없다!. 랜덤 벡터로 부터 생성된 이미지만 확인하면 되므로
    ## 하지만 pix2pix는 G에 condition image가 들어가고 해당 이미지와 real-image가 pair로 존재하기 때문에 val이 가능하다.
    
    result_dir_train = os.path.join(result_dir, 'train')
    result_dir_val = os.path.join(result_dir, 'val')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'png'))
    if not os.path.exists(result_dir_val):
        os.makedirs(os.path.join(result_dir_val, 'png'))

    # 데이터 부르기
    if mode == 'train':
        transform_train = transforms.Compose([Resize(shape=(286,286,nch)), RandomCrop((ny,nx))
        , Normalization(mean=0.5,std=0.5)]) # jitter technic이 사용됨(data augumentation)
                                            # Random jitter was applied by resizing the 256×256 input
                                            # images to 286 × 286 and then randomly cropping back to size 256 × 256.
        dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'), transform=transform_train, task=task, opts=opts)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        transform_val = transforms.Compose([Resize(shape=(286,286,nch)), RandomCrop((ny,nx))
        , Normalization(mean=0.5,std=0.5)]) 
        dataset_val = Dataset(data_dir=os.path.join(data_dir,'val'), transform=transform_train, task=task, opts=opts)
        loader_val = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8) # DateLoader는 Dataset class의 instance를 인자로 받아 데이터를 batch_size만큼씩 
                                                                                                   # yield 하는 generator를 만든다.


        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)
        num_data_val = len(dataset_val)
        num_batch_val = np.ceil(num_data_val / batch_size)

    ## 네트워크 생성하기
    if network == "pix2pix":
        netG = Pix2Pix_generator(in_channels=nch,out_channels=nch,nker=nker).to(device) # G는 input과 동일한 채널의 아웃풋 이미를 생성
        netD = Pix2Pix_Discriminator(in_channels=2 * nch,out_channels=1,nker=nker).to(device) # D에는 concat(G(x),x) 이므로 인풋 채널 2배 

        # 네트워크 가중치를 정규분포, std 0.02로 초기화
        init_weights(netG,init_type='normal',init_gain=0.02)
        init_weights(netD,init_type='normal',init_gain=0.02)

    ## 손실함수 정의하기
    loss_L1 = nn.L1Loss().to(device)
    loss_gan = nn.BCELoss().to(device) # D는 real, fake 이진분류, 따라서 binary cross entropy

    ## Optimizer 설정하기
    ## 하나의 network에 하나의 optimizer가 필요, GAN은 네트워크가 두개 이므로 optimizer도 두개 필요
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5,0.999)) 
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

            loss_G_L1_train= []
            loss_G_gan_train = []
            loss_D_real_train = []
            loss_D_fake_train = [] # D(G(z))

            for batch, data in enumerate(loader_train, 1): # loader_train에는 input, label pair data.
                # forward pass
                label = data['label'].to(device) # label : 실제이미지 data, device로 올리기 > D의 real 파트 인풋 y
                input = data['input'].to(device) # input : G의 condition 입력 이미지 data, x 
                output = netG(input) # G(x)

                # backward pass
                # backprop도 두 네트워크 각각 해줘야 한다.

                # Discriminator backprop
                set_requires_grad(netD,True) # netD(Discriminator)의 모든 파라미터 연산을 추적해 gradient를 계산한다.
                optimD.zero_grad() # gradient 초기화

                real = torch.cat([input,label], dim=1)
                fake = torch.cat([output,input],dim=1)

                pred_real = netD(real)  # D(x,y)
                pred_fake = netD(fake.detach()) # D(G(x),x) , detach는 D의 gradient가 G까지 흘러가지 않게 하기위해서

                loss_D_real = loss_gan(pred_real,torch.ones_like(pred_real)) 
                loss_D_fake = loss_gan(pred_fake,torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_fake + loss_D_real)

                loss_D.backward() # gradient backprop
                optimD.step() # gradient update

                # Generator backprop
                set_requires_grad(netD,False) # generator를 학습할땐, discriminator는 고정한다. 따라서 required_grad = False
                optimG.zero_grad() # gradient 초기화

                fake = torch.cat([input,output],dim = 1) 
                pred_fake = netD(fake) # D(G(x),x)

                loss_G_gan = loss_gan(pred_fake,torch.ones_like(pred_fake))  # generator는 생성한 가짜 이미지가 진짜(1)로 분류되게 학습해야한다.
                loss_G_L1 = loss_L1(output, label) # G(x), y의 L1 loss , 생성된 이미지와 real-image 사이의 유클리디안 거리
                loss_G = loss_G_gan + wgt * loss_G_L1

                loss_G.backward() # gradient backprop
                optimG.step() # gradient update


                # loss check
                loss_G_L1_train += [loss_G_L1.item()]
                loss_G_gan_train += [loss_G_gan.item()]
                loss_D_real_train += [loss_D_real.item()]
                loss_D_fake_train += [loss_D_fake.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | G_L1_LOSS %.4f |G_gan_LOSS %.4f  | D_fake_LOSS %.4f | D_real_LOSS %.4f " %
                    (epoch, num_epoch, batch, num_batch_train, np.mean(loss_G_L1_train),np.mean(loss_G_gan_train), np.mean(loss_D_fake_train),np.mean(loss_D_real_train)))

                if batch % 10 == 0:
                # 결과물 저장
                    input = fn_tonumpy(fn_denorm(input,mean=0.5,std=0.5)).squeeze() # generator로 생성된 output은 마지막 layer에 tanh를 거치며 normalize된다.
                    label = fn_tonumpy(fn_denorm(label,mean=0.5,std=0.5)).squeeze() # 따라서 denorm을 통해 복원
                    output = fn_tonumpy(fn_denorm(output,mean=0.5,std=0.5)).squeeze()
                                                                                
                    id = num_batch_train * (epoch - 1) + batch

                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)

                    writer_train.add_image('input', output, id, dataformats='NHWC')
                    writer_train.add_image('label', output, id, dataformats='NHWC')
                    writer_train.add_image('output', output, id, dataformats='NHWC')

            writer_train.add_scalar('loss_G_L1', np.mean(loss_G_L1_train), epoch)
            writer_train.add_scalar('loss_G_gan', np.mean(loss_G_gan_train), epoch)
            writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)
            writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)

            # val
            with torch.no_grad():
                netG.eval()
                netD.eval() 

                loss_G_L1_val= []
                loss_G_gan_val = []
                loss_D_real_val = []
                loss_D_fake_val = [] 

                for batch, data in enumerate(loader_val, 1): 
                    # forward pass
                    label = data['label'].to(device) # label : 실제이미지 data, device로 올리기 > D의 real 파트 인풋 y
                    input = data['input'].to(device) # input : G의 condition 입력 이미지 data, x 
                    output = netG(input) # G(x)

                    # val에서 backpropa는 하지 않고 학습된 네트워크에 val 데이터를 넣어 loss만 확인한다
                    real = torch.cat([input,label], dim=1)
                    fake = torch.cat([output,input],dim=1)

                    pred_real = netD(real)  # D(x,y)
                    pred_fake = netD(fake) # D(G(x),x) 

                    loss_D_real = loss_gan(pred_real,torch.ones_like(pred_real))
                    loss_D_fake = loss_gan(pred_fake,torch.zeros_like(pred_fake))
                    loss_D = 0.5 * (loss_D_fake + loss_D_real)


                    loss_G_gan = loss_gan(pred_fake,torch.ones_like(pred_fake))  # generator는 생성한 가짜 이미지가 진짜(1)로 분류되게 학습해야한다.
                    loss_G_L1 = loss_L1(output, label) # G(x), y의 L1 loss , 생성된 이미지와 real-image 사이의 유클리디안 거리
                    loss_G = loss_G_gan + wgt * loss_G_L1

                
                    # loss check
                    loss_G_L1_val += [loss_G_L1.item()]
                    loss_G_gan_val += [loss_G_gan.item()]
                    loss_D_real_val += [loss_D_real.item()]
                    loss_D_fake_val += [loss_D_fake.item()]

                    print("Vaild: EPOCH %04d / %04d | BATCH %04d / %04d | G_L1_LOSS %.4f |G_gan_LOSS %.4f  | D_fake_LOSS %.4f | D_real_LOSS %.4f " %
                        (epoch, num_epoch, batch, num_batch_val, np.mean(loss_G_L1_val),np.mean(loss_G_gan_val), np.mean(loss_D_fake_val),np.mean(loss_D_real_val)))

                    if batch % 10 == 0:
                    # 결과물 저장
                        input = fn_tonumpy(fn_denorm(input,mean=0.5,std=0.5)).squeeze() # generator로 생성된 output은 마지막 layer에 tanh를 거치며 normalize된다.
                        label = fn_tonumpy(fn_denorm(label,mean=0.5,std=0.5)).squeeze() # 따라서 denorm을 통해 복원
                        output = fn_tonumpy(fn_denorm(output,mean=0.5,std=0.5)).squeeze()
                                                                                    
                        id = num_batch_train * (epoch - 1) + batch

                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_input.png' % id), input[0].squeeze(), cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_label.png' % id), label[0].squeeze(), cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)

                        writer_val.add_image('input', output, id, dataformats='NHWC')
                        writer_val.add_image('label', output, id, dataformats='NHWC')
                        writer_val.add_image('output', output, id, dataformats='NHWC')

                writer_val.add_scalar('loss_G_L1', np.mean(loss_G_L1_val), epoch)
                writer_val.add_scalar('loss_G_gan', np.mean(loss_G_gan_val), epoch)
                writer_val.add_scalar('loss_D_fake', np.mean(loss_D_fake_val), epoch)
                writer_val.add_scalar('loss_D_real', np.mean(loss_D_real_val), epoch)



            if epoch % 10 == 0: # epoch 2번마다 네트워크 checkpoint 저장
                save(ckpt_dir=ckpt_dir, netG=netG, netD = netD, optimG=optimG, optimD=optimD, epoch=epoch)

        writer_train.close() # tensorboard 닫기
        writer_val.close()


def test(args):
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

    wgt = args.wgt

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
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png'))


    
    if mode == 'test':
        transform_test = transforms.Compose([Resize(shape=(ny,nx,nch)), Normalization(mean=0.5,std=0.5)]) # generator output인 tanh(-1~1) 값과
                                                                                                          # D input scale을 맞춰주기 위해 normalize
        dataset_test = Dataset(data_dir=os.path.join(data_dir,'test'), transform=transform_test, task=task, opts=opts)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)



    ## 네트워크 생성하기
    if network == "pix2pix":
        netG = Pix2Pix_generator(in_channels=nch,out_channels=nch,nker=nker).to(device)
        netD = Pix2Pix_Discriminator(in_channels=2 * nch,out_channels=1,nker=nker).to(device)

        # 네트워크 가중치를 정규분포, std 0.02로 초기화
        init_weights(netG,init_type='normal',init_gain=0.02)
        init_weights(netD,init_type='normal',init_gain=0.02)

    ## 손실함수 정의하기
    fn_gan = nn.BCELoss().to(device)
    fn_L1 = nn.L1Loss().to(device)

    ## Optimizer 설정하기
    ## 하나의 network에 하나의 optimizer가 필요, GAN은 네트워크가 두개 이므로 optimizer도 두개 필요
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5,0.999)) 
    optimD = torch.optim.Adam(netD.parameters(), lr=lr,betas=(0.5,0.999))


    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)
    cmap = None

    
    st_epoch = 0

    # TRAIN MODE
    if mode == 'test':
      
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD = netD, optimG=optimG, optimD = optimD)

        # test에서는 D가 필요없이 G로 생성되는 이미지만 확인하면 된다.
        with torch.no_grad():
            netG.eval()
            
            loss_L1_test= []      
          
            for batch, data in enumerate(loader_test, 1): 
                # forward pass
                label = data['label'].to(device) # label : 실제이미지 data, device로 올리기 > D의 real 파트 인풋 y
                input = data['input'].to(device) # input : G의 condition 입력 이미지 data, x 
                output = netG(input) # G(x)

            
                # loss check
                loss_L1 = fn_L1(output, label)
                loss_L1_test += [loss_L1.item()]

                print("Test | BATCH %04d / %04d | G_L1_LOSS %.4f" %(batch, num_batch_test, np.mean(loss_L1)))

            
                input = fn_tonumpy(fn_denorm(input,mean=0.5,std=0.5)).squeeze()
                label = fn_tonumpy(fn_denorm(label,mean=0.5,std=0.5)).squeeze()
                output = fn_tonumpy(fn_denorm(output,mean=0.5,std=0.5)).squeeze()

                for j in range(label.shape[0]):
                             
                    id = batch_size * (batch - 1) + j

                    input_ = input[j]
                    label_ = label[j]
                    output_ = output[j]

                    plt.imsave(os.path.join(result_dir_test,'png','%04d_input.png'%id),input_.squeeze(), cmap = cmap)
                    plt.imsave(os.path.join(result_dir_test,'png','%04d_input.png'%id),label_.squeeze(), cmap = cmap)
                    plt.imsave(os.path.join(result_dir_test,'png','%04d_input.png'%id),output_.squeeze(), cmap = cmap)



            print("Average L1 Loss : %04d" % np.mean(loss_L1_test))