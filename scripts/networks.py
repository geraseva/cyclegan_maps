# -*- coding: cp1251 -*-
import torch
import torch.nn as nn
from torch.nn import init

# опишем генератор и дискриминатор 

class Generator(nn.Module):
    def __init__(self, num_blocks=6):
        super(Generator,self).__init__()
        self.num_blocks=num_blocks
        model=[nn.ReflectionPad2d(6),
               nn.Conv2d(3, 64, kernel_size=7, padding=0),
               nn.BatchNorm2d(64),
               nn.ReLU(True),
               nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
               nn.BatchNorm2d(128),
               nn.ReLU(True),
               nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
               nn.BatchNorm2d(256),
               nn.ReLU(True)]
        for i in range(self.num_blocks):
            model += [ResnetBlock()]
        model += [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, 
                                     padding=0, output_padding=1),
                  nn.BatchNorm2d(128),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, 
                                     padding=0, output_padding=1),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True), 
                  nn.Conv2d(64, 3, kernel_size=7, padding=0), 
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock,self).__init__()
        conv_block=[nn.ReflectionPad2d(2),
                    nn.Conv2d(256, 256, kernel_size=3), 
                    nn.BatchNorm2d(256), 
                    nn.ReLU(True),
                    #nn.ReflectionPad2d(1),
                    nn.Conv2d(256, 256, kernel_size=3), 
                    nn.BatchNorm2d(256)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), 
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                 nn.BatchNorm2d(128),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                 nn.BatchNorm2d(256),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                 nn.BatchNorm2d(512),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)] 
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
