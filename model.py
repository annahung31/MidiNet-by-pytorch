import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ops import *


class sample_generator(nn.Module):
    def __init__(self):
        super(sample_generator, self).__init__()
        self.gf_dim   = 64
        self.y_dim   = 13
        self.n_channel = 256

        self.h1      = nn.ConvTranspose2d(in_channels=157, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h2      = nn.ConvTranspose2d(in_channels=157, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h3      = nn.ConvTranspose2d(in_channels=157, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h4      = nn.ConvTranspose2d(in_channels=157, out_channels=1, kernel_size=(1,pitch_range), stride=(1,2))

        self.h0_prev = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,pitch_range), stride=(1,2))
        self.h1_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))
        self.h2_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))
        self.h3_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))

        self.linear1 = nn.Linear(113,1024)
        self.linear2 = nn.Linear(1037,self.gf_dim*2*2*1)

    def forward(self, z, prev_x, y ,batch_size,pitch_range):

        # h3_prev = F.leaky_relu(self.batch_nor_256(self.h0_prev(prev_x)),0.2)
        h0_prev = lrelu(batch_norm_2d_cpu(self.h0_prev(prev_x)),0.2)   #[72, 16, 16, 1]
        h1_prev = lrelu(batch_norm_2d_cpu(self.h1_prev(h0_prev)),0.2)  #[72, 16, 8, 1]
        h2_prev = lrelu(batch_norm_2d_cpu(self.h2_prev(h1_prev)),0.2)  #[72, 16, 4, 1]
        h3_prev = lrelu(batch_norm_2d_cpu(self.h3_prev(h2_prev)),0.2)  #[72, 16, 2, 1])

        yb = y.view(batch_size,  self.y_dim, 1, 1)  #(72,13,1,1)

        z = torch.cat((z,y),1)         #(72,113)

        h0 = F.relu(batch_norm_1d_cpu(self.linear1(z)))    #(72,1024)
        h0 = torch.cat((h0,y),1)   #(72,1037)

        h1 = F.relu(batch_norm_1d_cpu(self.linear2(h0)))   #(72, 256)
        h1 = h1.view(batch_size, self.gf_dim * 2, 2, 1)     #(72,128,2,1)
        h1 = conv_cond_concat(h1,yb) #(b,141,2,1)
        h1 = conv_prev_concat(h1,h3_prev)  #(72, 157, 2, 1)

        h2 = F.relu(batch_norm_2d_cpu(self.h1(h1)))  #(72, 128, 4, 1)
        h2 = conv_cond_concat(h2,yb) #([72, 141, 4, 1])
        h2 = conv_prev_concat(h2,h2_prev)  #([72, 157, 4, 1])

        h3 = F.relu(batch_norm_2d_cpu(self.h2(h2)))  #([72, 128, 8, 1]) 
        h3 = conv_cond_concat(h3,yb)  #([72, 141, 8, 1])
        h3 = conv_prev_concat(h3,h1_prev) #([72, 157, 8, 1])

        h4 = F.relu(batch_norm_2d_cpu(self.h3(h3)))  #([72, 128, 16, 1])
        h4 = conv_cond_concat(h4,yb)  #([72, 141, 16, 1])
        h4 = conv_prev_concat(h4,h0_prev) #([72, 157, 16, 1])

        g_x = torch.sigmoid(self.h4(h4)) #([72, 1, 16, 128])

        return g_x



class generator(nn.Module):
    def __init__(self,pitch_range):
        super(generator, self).__init__()
        self.gf_dim   = 64
        self.y_dim   = 13
        self.n_channel = 256

        self.h1      = nn.ConvTranspose2d(in_channels=157, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h2      = nn.ConvTranspose2d(in_channels=157, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h3      = nn.ConvTranspose2d(in_channels=157, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h4      = nn.ConvTranspose2d(in_channels=157, out_channels=1, kernel_size=(1,pitch_range), stride=(1,2))

        self.h0_prev = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,pitch_range), stride=(1,2))
        self.h1_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))
        self.h2_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))
        self.h3_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))

        self.linear1 = nn.Linear(113,1024)
        self.linear2 = nn.Linear(1037,self.gf_dim*2*2*1)

    def forward(self, z, prev_x, y ,batch_size,pitch_range):

        # h3_prev = F.leaky_relu(self.batch_nor_256(self.h0_prev(prev_x)),0.2)
        h0_prev = lrelu(batch_norm_2d(self.h0_prev(prev_x)),0.2)   #[72, 16, 16, 1]
        h1_prev = lrelu(batch_norm_2d(self.h1_prev(h0_prev)),0.2)  #[72, 16, 8, 1]
        h2_prev = lrelu(batch_norm_2d(self.h2_prev(h1_prev)),0.2)  #[72, 16, 4, 1]
        h3_prev = lrelu(batch_norm_2d(self.h3_prev(h2_prev)),0.2)  #[72, 16, 2, 1])

        yb = y.view(batch_size,  self.y_dim, 1, 1)  #(72,13,1,1)

        z = torch.cat((z,y),1)         #(72,113)

        h0 = F.relu(batch_norm_1d(self.linear1(z)))    #(72,1024)
        h0 = torch.cat((h0,y),1)   #(72,1037)

        h1 = F.relu(batch_norm_1d(self.linear2(h0)))   #(72, 256)
        h1 = h1.view(batch_size, self.gf_dim * 2, 2, 1)     #(72,128,2,1)
        h1 = conv_cond_concat(h1,yb) #(b,141,2,1)
        h1 = conv_prev_concat(h1,h3_prev)  #(72, 157, 2, 1)

        h2 = F.relu(batch_norm_2d(self.h1(h1)))  #(72, 128, 4, 1)
        h2 = conv_cond_concat(h2,yb) #([72, 141, 4, 1])
        h2 = conv_prev_concat(h2,h2_prev)  #([72, 157, 4, 1])

        h3 = F.relu(batch_norm_2d(self.h2(h2)))  #([72, 128, 8, 1]) 
        h3 = conv_cond_concat(h3,yb)  #([72, 141, 8, 1])
        h3 = conv_prev_concat(h3,h1_prev) #([72, 157, 8, 1])

        h4 = F.relu(batch_norm_2d(self.h3(h3)))  #([72, 128, 16, 1])
        h4 = conv_cond_concat(h4,yb)  #([72, 141, 16, 1])
        h4 = conv_prev_concat(h4,h0_prev) #([72, 157, 16, 1])

        g_x = torch.sigmoid(self.h4(h4)) #([72, 1, 16, 128])

        return g_x


class discriminator(nn.Module):
    def __init__(self,pitch_range):
        super(discriminator, self).__init__()

        self.df_dim = 64
        self.dfc_dim = 1024
        self.y_dim = 13

        self.h0_prev = nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(2,pitch_range), stride=(2,2))
        #out channels = y_dim +1 
        self.h1_prev = nn.Conv2d(in_channels=27, out_channels=77, kernel_size=(4,1), stride=(2,2))
        # out channels = df_dim + y_dim
        self.linear1 = nn.Linear(244,self.dfc_dim)
        self.linear2 = nn.Linear(1037,1)

    def forward(self,x,y,batch_size,pitch_range):        

        yb = y.view(batch_size,self.y_dim, 1, 1)
        x = conv_cond_concat(x, yb)  #x.shape torch.Size([72, 14, 16, 128])
        
        h0 = lrelu(self.h0_prev(x),0.2)
        fm = h0
        h0 = conv_cond_concat(h0, yb) #torch.Size([72, 27, 8, 1])

        h1 = lrelu(batch_norm_2d(self.h1_prev(h0)),0.2)  #torch.Size([72, 77, 3, 1])
        h1 = h1.view(batch_size, -1)  #torch.Size([72, 231])
        h1 = torch.cat((h1,y),1)  #torch.Size([72, 244])

        h2 = lrelu(batch_norm_1d(self.linear1(h1)))
        h2 = torch.cat((h2,y),1)  #torch.Size([72, 1037])

        h3 = self.linear2(h2)
        h3_sigmoid = torch.sigmoid(h3)


        return h3_sigmoid, h3, fm

