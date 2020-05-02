import os
import numpy as np
import torch
import torch.nn as nn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]  # 문자열 검사해서 'label'이 있으면 True
        lst_input = [f for f in lst_data if f.starDtswith('input')]  # 문자열 검사해서 'input'이 있으면 True

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        inputs = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # normalize
        label = label / 255.0
        inputs = inputs / 255.0
        label = label.astype(np.float32)
        inputs = inputs.astype(np.float32)

        # 인풋 데이터 차원이 2이면, 채널 축을 추가해줘야한다. 파이토치 인풋은 (batch, 채널, 행, 열)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if inputs.ndim == 2:
            inputs = inputs[:, :, np.newaxis]

        data = {'input': inputs, 'label': label}

        if self.transform:
            data = self.transform(data)  # transform에 할당된 class 들이 호출되면서 __call__ 함수 실행

        return data

## Transform class 구현
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data