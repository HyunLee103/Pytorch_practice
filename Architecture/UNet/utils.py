import os
import numpy as np
import torch
import torch.nn as nn


# 네트워크 저장하기
# train을 마친 네트워크 저장
# net (네트워크 파라미터), optim  두개를 dict 형태로 저장
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, '%s/model_epoch%d.pth' % (ckpt_dir, epoch))


# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):  # 저장된 네트워크가 없다면 인풋을 그대로 반환
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)  # ckpt_dir 아래 있는 모든 파일 리스트를 받아온다
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch