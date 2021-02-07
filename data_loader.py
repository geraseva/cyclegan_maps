# -*- coding: cp1251 -*-

import torch
import torch.nn as nn
from torch.nn import init

from torchvision import transforms
from torch.utils.data import Dataset

from skimage.io import imread

class Monet2PhotoDataset(Dataset):
    def __init__(self, files):
        super(Monet2PhotoDataset).__init__()
        self.len_ = len(files)
        self.files = sorted(files)
    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        tr=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.RandomHorizontalFlip()])
        x=imread(self.files[index])
        #x=np.rollaxis(x ,2, 0)
        x=tr(x)
        return x
