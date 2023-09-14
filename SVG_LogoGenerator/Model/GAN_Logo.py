# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 17:04:01 2021

@author: ChangGun Choi
"""

!pip3 install svglib
#%%
from svglib.svglib import svg2rlg  # https://stackoverflow.com/questions/6589358/convert-svg-to-png-in-python
from reportlab.graphics import renderPM
import os

#'C:/Users/ChangGun Choi/Team Project/file/filelist_svg[0]' + '.png'

# SVG
filelist_svg= [file.rsplit('.', 1)[0] for file in os.listdir('C:/Users/ChangGun Choi/Team Project/svg_file') if file.endswith('.svg')]
len(filelist_svg)

for s in filelist_svg:
    drawing = svg2rlg('C:/Users/ChangGun Choi/Team Project/svg_file/' + s + '.svg' )
    renderPM.drawToFile(drawing, 'C:/Users/ChangGun Choi/Team Project/file/' + s + '.png', fmt='PNG')

# PNG
filelist_png= [file for file in os.listdir('C:/Users/ChangGun Choi/Team Project/png_file') if file.endswith('.png')]
len(filelist_png)

#%%
# Use GPU if CUDA is available
DATA_PATH =  "C:/Users/ChangGun Choi/Team Project/png_file"
MODEL_PATH = "C:/Users/ChangGun Choi/Team Project/model"
torch.cuda.is_available()  
#DEVICE = 'cpu'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
import os
import sys
import torch
import torch.nn as nn

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from helper import *                  #importing helper.py
#from util import nextplot   
from PIL import Image 
from matplotlib.pyplot import imshow

img = Image.open('C:/Users/ChangGun Choi/Team Project/png_file/7-eleven.png') 
imshow(np.asarray(img))


#%%
transforms_train = transforms.Compose([
    transforms.Resize(64),  # 28 * 28
    transforms.ToTensor(),
    transforms.CenterCrop(64),  ### necessary
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # channel 3 (RGB)
])
path = "C:/Users/ChangGun Choi/Team Project/file" 
train_dataset = datasets.ImageFolder(root=path, transform=transforms_train)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
                                                                    # Randomly sample batch
train_dataset.__getitem__(0)
len(train_dataset)
#%%
for i, data in enumerate(dataloader):
    print(data[0].size())  # input image     # torch.Size([4, 3, 28, 28]) (batch_size, RGB, pixel, pixel)
    #print(data[1])         # class label

for i, (imgs, labels) in enumerate(dataloader):    # What is label of PNG file ?????
  print(imgs.size())
  #print(labels)    # tensor([0, 0, 0, 0])
 
#%%  GAN
latent_dim = 100      # dimension for latent vector "Z"


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Defining 1 Block
        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]          # Linear layer???
            if normalize: 
                # batch normalization -> Same dimension
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Many blocks are sequentially stacked
        self.model = nn.Sequential(                   # each block -> what is * ????
            *block(latent_dim, 128, normalize=False), # (Z dimension, output_dim) 
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 3 * 64 * 64),            # Flatten: Fully-Connteced layer 
            nn.Tanh()                                
        )

    def forward(self, z):
        img = self.model(z)                          # Generater_model: latent Z  input    
        img = img.view(img.size(0), 3, 64, 64)       
        return img                                    
    
#%%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3 * 64 * 64, 512),   
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),             
            nn.Sigmoid(), # classification 0 ~ 1
        )

    # Give back result of discrimation
    def forward(self, img):
        flattened = img.view(img.size(0), -1) 
        output = self.model(flattened)

        return output
    

#%%
# Initialization
generator = Generator()
discriminator = Discriminator()

generator.cuda()
discriminator.cuda()

# Loss Function
adversarial_loss = nn.BCELoss()
adversarial_loss.cuda()

# learning rate
lr = 0.0002

# Optimzation
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Visualize
fixed_noise = torch.randn(4, 100, 1, 1, device=DEVICE)
#%%
import time

n_epochs = 300 # How many epochs
sample_interval = 100 # Which interval of Batch, want to see the results
start_time = time.time()

for epoch in range(n_epochs):
    for i, (imgs, label) in enumerate(dataloader):

        # Creating Ground_truth Label 
        real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0) # real: 1   # imgs.size(0) : Batch Size 
        fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0) # fake: 0

        real_imgs = imgs.cuda()    

        """ Train generator """
        optimizer_G.zero_grad()

        # Sampling random noise
        z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()  # (input_d , latent_dim)
                                                                                  # imgs.shape[0] ?????
        # Generate image
        generated_imgs = generator(z)

        # Loss calculate
        g_loss = adversarial_loss(discriminator(generated_imgs), real)

        # generator update
        g_loss.backward()
        optimizer_G.step()

        """ Train discriminator """
        optimizer_D.zero_grad()

        # discriminator loss
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # update
        d_loss.backward()
        optimizer_D.step()

        done = epoch * len(dataloader) + i
        if done % sample_interval == 0:
            # Show only 25 images from generated pics
            save_image(generated_imgs.data[:25], f"{done}.png", nrow=5, normalize=True)

    # print log each epoch
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]")
    

#%%

from IPython.display import Image

Image('.png')