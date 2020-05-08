import os
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from util import *

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        lst_data.sort()
        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):  # iterator 만들기
        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        sz = img.shape

        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        if img.dtype == np.uint8:
            img = img / 255.0

        # 이미지의 label(y), input(x)을 결정해, 학습 방향을 정하는 옵션
        if self.opts[0] == 'direction':
            if self.opts[1] == 0: # label : left, input : right
                data = {'label':img[:, :sz[1]//2, :] ,'input':img[: , sz[1]//2:, :]}
            elif self.opts[1] == 1: # 반대
                data = {'label':img[:, sz[1]//2: ,:], 'input':img[:, :sz[1]//2,:]}
        else:
            data = {'label': img}

        if self.transform:
            data = self.transform(data)

        data = self.to_tensor(data)

        return data

## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data



# DCGAN에 사용할 selanA image data가 DCGAN 모델의 generator output인 64x64와 맞지 않으므로
# resize 해주는 transform class 선언
class Resize(object):
    def __init__(self,shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0],self.shape[1],
                                                    self.shape[2]))
        return data

class RandomCrop(object):
  def __init__(self, shape):
      self.shape = shape

  def __call__(self, data):
    # input, label = data['input'], data['label']
    # h, w = input.shape[:2]

    h, w = data['label'].shape[:2]
    new_h, new_w = self.shape

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
    id_x = np.arange(left, left + new_w, 1)

    # input = input[id_y, id_x]
    # label = label[id_y, id_x]
    # data = {'label': label, 'input': input}

    # Updated at Apr 5 2020
    for key, value in data.items():
        data[key] = value[id_y, id_x]

    return data



















