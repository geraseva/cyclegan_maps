# -*- coding: cp1251 -*-
"""gan.ipynb

Ну типа, у нас 4 сети: G, F, DG и DF. Первые 2 это прямой и
обратный генераторы, а последние 2 - соответствующие
дискриминаторы. Опишем шаг обучения: в сайкл ган подается 2
батча (x и y) из 2 соответствующих датасетов. Они грузятся
в соответствующие генераторы, и выдачи записываются:
G: x -> y_fake, F: y -> x_fake. Фейки грузятся в
комплементарные генераторы для восстановления
(G: x_fake -> y_rec, F: y_fake -> x_rec). Считается лосс как
сумма cycle loss (видимо, это просто сумма матожиданий
x_rec-x и y_rec-y) и gan loss (он считается из выдачи
дискриминаторов DG(y_fake) и DF(x_fake)). В коде авторов был
еще какой-то добавочный член и какие-то непонятные
коэффициенты, но это потом. После подсчета лосса считаются
производные и пересчитываются веса генераторов (не забыть
заморозить веса дискриминаторов). Далее считаются лоссы для
дискриминаторов (как среднее от gan loss для фейковых и
исходных батчей), и пересчитываются уже их веса (почему-то в
коде авторов при этом забывают заморозить веса генераторов,
хотя теоретически надо бы; насколько я понимаю, оптимизатору
там заранее подаются на вход конкретные веса, и он работает
только с ними).
"""

import torch
import torch.nn as nn
from torch.nn import init

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler

from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import clear_output

import itertools
import os
import math
import numpy as np
from time import time

import argparse
from networks import Generator, Discriminator
from data_loader import SimpleDataset, CityMapDataset

import warnings
from PIL import Image

warnings.simplefilter('ignore', Image.DecompressionBombWarning) 


parser = argparse.ArgumentParser()
parser.add_argument('--trainA', type=str, help='path to first training dataset',
                    default='../datasets/trainA')
parser.add_argument('--trainB', type=str, help='path to another training dataset',
                    default='../datasets/trainB')
parser.add_argument('--epochs', type=int, help='number of training epochs',
                    default=1)
parser.add_argument('--weights', help='pre-counted weights for the model',
                    default=None)
parser.add_argument('--save_weights', help='where to save weights',
                    default='../weights')
parser.add_argument('--datasize', type=int, help='size of a training dataset for 1 epoch',
                    default=1000)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Process is working on', device, flush=True)


A_files=[]
B_files=[]

for img in os.listdir(args.trainA):
    A_files.append(os.path.join(args.trainA, img))
for img in os.listdir(args.trainB):
    B_files.append(os.path.join(args.trainB, img))
    
print('Datasets lengths are %d and %d respectively' % (len(A_files),len(B_files)))

A_data=SimpleDataset(A_files)
B_data=SimpleDataset(B_files)

# создаем генераторы и дискриминаторы
g = Generator().to(device)
f = Generator().to(device)
dg = Discriminator().to(device)
df = Discriminator().to(device)

# инициализируем веса, если есть
if args.weights != None:
    g.load_state_dict(torch.load(os.path.join(args.weights,'g_weights.pt')))
    f.load_state_dict(torch.load(os.path.join(args.weights,'f_weights.pt')))
    dg.load_state_dict(torch.load(os.path.join(args.weights,'dg_weights.pt')))
    df.load_state_dict(torch.load(os.path.join(args.weights,'df_weights.pt')))


cycle_opt=torch.optim.Adam(itertools.chain(g.parameters(), f.parameters()), lr=0.0002)
gan_opt=torch.optim.Adam(itertools.chain(dg.parameters(), df.parameters()), lr=0.0002)

gan_crit=nn.BCEWithLogitsLoss()
cycle_crit=nn.L1Loss()

# шаг обучения будет выглядеть как-то так:
torch.autograd.set_detect_anomaly(True)
def train_step(x, y, batch_size=1):
    x=x.to(device,dtype=torch.float)
    y=y.to(device,dtype=torch.float)
    # сначала прогоняем батчи через генераторы туда и обратно
    y_fake=g(x)
    x_fake=f(y)
    x_rec=f(y_fake)
    y_rec=g(x_fake)
    g_qual=dg(y_fake)
    f_qual=df(x_fake)
    
    # морозим дискриминаторы
    for par in itertools.chain(dg.parameters(), df.parameters()):
        par.requires_grad=False    

    # считаем лосс для генераторов  
    cycle_loss=cycle_crit(x_rec, x)+cycle_crit(y_rec, y) \
    +gan_crit(g_qual, torch.ones(g_qual.shape).to(device)) \
    +gan_crit(f_qual, torch.ones(f_qual.shape).to(device))
    cycle_opt.zero_grad()
    # считаем производные
    cycle_loss.backward(retain_graph=True)
    # обновляем веса
    cycle_opt.step()
    # тут оптимизируем дискриминаторы
    for par in itertools.chain(dg.parameters(), df.parameters()):
        par.requires_grad=True    

    g_qual=dg(y_fake.detach())
    gan_opt.zero_grad()
    gan_loss_dg=0.5*(gan_crit(dg(y), torch.ones(g_qual.shape).to(device)) \
                     +gan_crit(g_qual, torch.zeros(g_qual.shape).to(device)))
    gan_loss_dg.backward()

    f_qual=df(x_fake.detach())
    gan_loss_df=0.5*(gan_crit(df(x), torch.ones(f_qual.shape).to(device)) \
                     +gan_crit(f_qual, torch.zeros(f_qual.shape).to(device)))
    gan_loss_df.backward()
    gan_opt.step()

    return float(cycle_loss), 0.5*(float(gan_loss_dg)+float(gan_loss_df))

f.train()
g.train()

for epoch in range(args.epochs):
    t0=time()
    if args.datasize==len(A_files):
        A=DataLoader(A_data, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    else:
        A_sampler=RandomSampler(A_data, replacement = True,
                            num_samples = args.datasize)
        A=DataLoader(A_data, batch_size=1, sampler=A_sampler, num_workers=2, pin_memory=True)
    if args.datasize==len(B_files):
        B=DataLoader(B_data, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    else:
        B_sampler=RandomSampler(B_data, replacement = True,
                            num_samples = args.datasize)
        B=DataLoader(B_data, batch_size=1, sampler=B_sampler, num_workers=2, pin_memory=True)
    cl=0
    gl=0
    print('Epoch %d :' % epoch)
    i=0
    for a_pic, b_pic in zip(A, B):
        loss1, loss2=train_step(a_pic, b_pic)
        cl += loss1/args.datasize
        gl += loss2/args.datasize
        i+=1
    #    if i>1000:
    #        break
    t=time()-t0
    print('Cycle loss: %f' % cl)
    print('GAN loss: %f' % gl)
    print('Time: %f seconds' % t, flush=True)
    
    #Я сочла нужным сохранять веса на каждой эпохе, 
    #на случай прерывания программы
    torch.save(g.state_dict(),os.path.join(args.save_weights, 'g_weights.pt'))
    torch.save(f.state_dict(),os.path.join(args.save_weights, 'f_weights.pt'))
    torch.save(dg.state_dict(),os.path.join(args.save_weights, 'dg_weights.pt'))
    torch.save(df.state_dict(),os.path.join(args.save_weights, 'df_weights.pt'))

    


