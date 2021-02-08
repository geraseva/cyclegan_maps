# -*- coding: cp1251 -*-

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset

from skimage.io import imread
from skimage.transform import resize
from skimage.util import crop
from skimage.color import rgba2rgb

from random import randint

class SimpleDataset(Dataset):
    def __init__(self, files):
        super(SimpleDataset).__init__()
        self.len_ = len(files)
        self.files = sorted(files)
    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        tr=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip()])
        x=imread(self.files[index])
        #x=np.rollaxis(x ,2, 0)
        x=tr(x)
        return x
class CityMapDataset(Dataset):
    def __init__(self, files):
        super(CityMapDataset).__init__()
        self.len_ = len(files)
        self.files = sorted(files)
    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        
        x=imread(self.files[index]).astype(np.float)
        square=600
        h, w, c = x.shape
        if c>3:
            x=rgba2rgb(x)
        cropsize=randint(min(square,h,w),min(h,w))
        x=crop(x, ((h-cropsize,0),(w-cropsize,0),(0,0)))
        h, w, c = x.shape
        if h>w:
            x=crop(x, ((0,(h-w)),(0,0),(0,0)))
        elif w>h:
            x=crop(x, ((0,0),(0,(w-h)),(0,0)))
        else:
            pass
        x=resize(x, (square,square))
        tr=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip()])

        #x=np.rollaxis(x ,2, 0)
        x=tr(x)
        return x
