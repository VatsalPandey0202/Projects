# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:32:58 2021

@author: ChangGun Choi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 17:04:01 2021

@author: ChangGun Choi
"""
#%%
# Use GPU if CUDA is available
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
torch.cuda.is_available()  
#DEVICE = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu" 
torch.manual_seed(1)

img = Image.open('C:/Users/ChangGun Choi/Team Project/png_file/7-eleven.png') 
imshow(np.asarray(img))


#%%
transforms_train = transforms.Compose([
    transforms.Resize(64),  
    transforms.ToTensor(),  
    transforms.CenterCrop(64),  ### necessary
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)) # -1 ~ 1
])

#path = "C:/Users/ChangGun Choi/Team Project/file" 
#path = "C:/Users/ChangGun Choi/Desktop/0. 수업자료/3_videos_project/TEAM Project/SVG_LogoGenerator/Data"
train_dataset = datasets.ImageFolder(root=path, transform=transforms_train)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
                                                                    # Randomly sample batch
# drop_last : length of Dataset is not dividable by the batch size without a remainder, which happens to be 1  
                                                            
train_dataset.__getitem__(0)                                        
len(train_dataset)
#%% Example
for i, data in enumerate(dataloader):
    print(data[0].size())  # input image     # torch.Size([4, 3, 28, 28]) (batch_size, RGB, pixel, pixel)
    #print(data[1])         # class label

for i, (imgs, labels) in enumerate(dataloader):
  print(imgs.size())
  #print(labels)    # tensor([0, 0, 0, 0])
 
#%%  GAN https://wikidocs.net/62306 ,  https://dreamgonfly.github.io/blog/gan-explained/
# Generator Input : Latent vector z (batch_size x 100), Output: batch_size x 3*64*64 RGB
params = {'num_classes': 7,  # If class is 7 categories
          'nz':100,
          'input_size':(3,64,64)}

class Generator(nn.Module):                                   
    def __init__(self, params):
        super().__init__()
        self.num_classes = params['num_classes'] # 
        self.nz = params['nz'] # noise dimension, 100
        self.input_size = params['input_size'] # (3,64,64)
         # noise, label  -> label embedding matrix
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)  # Lookup table (num_classes=len(vocab), dimension)
        self.latent_dim = self.nz + self.num_classes      # dimension for latent vector "Z"
        
        # Defining 1 Block
        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]        
            if normalize:                                      #
                # batch normalization -> Same dimension
                layers.append(nn.BatchNorm1d(output_dim, 0.8)) # 0.8 ??
            layers.append(nn.LeakyReLU(0.2, inplace=True))   
            return layers

        # Many blocks are sequentially stacked        # MLP perceptron
        self.gen  = nn.Sequential(                   # each block -> what is * ????
            *block(self.latent_dim, 128, normalize=False), # (Z dimension, output_dim) 
            *block(128, 256),                 #
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 3 * 64 * 64),   
            nn.Tanh()              
            )
    def forward(self, noise, labels):
        # noise + label 
        gen_input = torch.cat((self.label_emb(labels),noise),-1)  #-1
        img = self.gen(gen_input)                            # Generater_model: latent Z 가 input    
        # Tanh 값에서 다시 가짜 이미지 생성하기
        img = img.view(img.size(0), 3, 64, 64)  
        return img                             

#%% Ex1)

x = torch.randn(16,100,device=device) # 
model_gen = Generator(params).to(device) # model
labels = torch.randint(0,10,(16,),device=device) #
out_gen = model_gen(x,labels) 
print(out_gen.shape)


labels     # torch.Size([16])
label_embedding = nn.Embedding(num_classes, num_classes).to(device) 
label_embedding   # tensor([5, 3, 9, 4, 6, 1, 6, 3, 0, 1, 3, 5, 6, 8, 8, 6])

label_embedding(labels).shape  # torch.Size([16, 10])
label_embedding(labels)

######
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["world"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
#%%

class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.input_size = params['input_size']
        self.num_classes = params['num_classes']
        self.label_embedding = nn.Embedding(self.num_classes, self.num_classes) 
        
        self.model = nn.Sequential(
            nn.Linear(self.num_classes + 3 * 64 * 64, 512),   
            nn.LeakyReLU(0.2 ,inplace=True),
            nn.Linear(512,512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1),
            nn.Sigmoid(), # classification 0 ~ 1
        )

    # Give back resqult of discrimation
    def forward(self, img, labels):               
        flattened = torch.cat((img.view(img.size(0),-1),self.label_embedding(labels)), dim =-1) # label : Embedding(16, 10)
        #flattened_1 = img.view(img.size(0), -1)  # 3 * 28 * 28 
        output = self.model(flattened)              
        return output

#%% check
x = torch.randn(16,3,64,64,device=device)
label = torch.randint(0,10,(16,), device=device)
model_dis = Discriminator(params).to(device)
out_dis = model_dis(x, label)
print(out_dis)

#%%
# Initialization
generator = Generator(params)
discriminator = Discriminator(params)
generator.cuda()
discriminator.cuda()

# Loss Function
adversarial_loss = nn.BCELoss()
adversarial_loss.cuda()

# learning rate
lr = 0.0002 #2e-4

# Optimzation
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Visualize
#fixed_noise = torch.randn(4, 100, 1, 1, device=DEVICE)
#%%

from PIL import Image 
Image.MAX_IMAGE_PIXELS = 1000000000 
import cv2

#https://stackoverflow.com/questions/21596281/how-does-one-convert-a-grayscale-image-to-rgb-in-opencv-python
#https://pypi.org/project/opencv-python/

def initialize_weights(model):
    classname = model.__class__.__name__
    # fc layer
    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    # batchnorm
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

generator.apply(initialize_weights);
discriminator.apply(initialize_weights);
#%%

generator.train()
discriminator.train()
nz = params['nz']
n_epochs = 10 # How many epochs
sample_interval = 1000 # Which interval of Batch, want to see the results
#start_time = time.time()
loss_history={'gen':[],
              'dis':[]}

for epoch in range(n_epochs):
    for i, (imgs, label) in enumerate(dataloader):

        #imgs = cv2.cvtColor(img, cv.CV_GRAY2RGB)  #####
        # Real image
        real_imgs = imgs.cuda()
        real_labels = label.cuda()  
        # Creating Ground_truth Label 
        real = torch.ones(imgs.size(0), 1).to(device)# real: 1   # imgs.size(0) : Batch Size 
        fake = torch.zeros(imgs.size(0), 1).to(device) # fake: 0

        """ Train generator """
        optimizer_G.zero_grad()
        # Sampling random noise
        noise = torch.normal(mean=0, std=1, size=(imgs.shape[0], 100)).cuda()  # (input_d , latent_dim)
        gen_label = torch.randint(0,7,(imgs.size(0),)).to(device) # random label  0~9                                                                # imgs.shape[0] ?????
        # Generate fake image
        generated_imgs = generator(noise, gen_label)
        # Discriminate fake image
        out_dis = discriminator(generated_imgs,gen_label)
        
        # Loss calculate
        g_loss = adversarial_loss(out_dis, real) # real : 1 
                                #[0.6,0.7,....]
        # generator update
        g_loss.backward()
        optimizer_G.step()
        ###########################################################################
        
        """ Train discriminator """
        optimizer_D.zero_grad()

        # discriminator loss
        real_loss = adversarial_loss(discriminator(real_imgs, real_labels), real)
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake) # not update G() during training D()
        # Detaching fake from the graph is necessary to avoid forward-passing the noise through G when we actually update the generator. 
        # If we do not detach, then, although fake is not needed for gradient update of D, it will still be added to the computational graph
        # and as a consequence of backward pass which clears all the variables in the graph (retain_graph=False by default), fake won't be available when G is updated.
        d_loss = (real_loss + fake_loss) / 2

        # update
        d_loss.backward()
        optimizer_D.step()
        
        loss_history['gen'].append(g_loss.item())
        loss_history['dis'].append(d_loss.item())
        
        done = epoch * len(dataloader) + i
        if done % sample_interval == 0:  
            # Show only 4 images from generated pics
            save_image(generated_imgs.data[:4], f"{done}.png", nrow=2, normalize=True)
           #nrows: column, #normalize: If True, shift the image to the range (0, 1)
           #https://aigong.tistory.com/183
    # print log each epoch
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]")
    
#%%
plt.figure(figsize=(10,5))
plt.title('Loss Progress')
plt.plot(loss_history['gen'], label='Gen. Loss')
plt.plot(loss_history['dis'], label='Dis. Loss')
plt.xlabel('batch count')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%  
path2models = 'C:/Users/ChangGun Choi/Team Project/model/'
os.makedirs(path2models, exist_ok=True)
path2weights_gen = os.path.join(path2models, 'weights_gen.pt')
path2weights_dis = os.path.join(path2models, 'weights_dis.pt')

torch.save(generator.state_dict(), path2weights_gen)
torch.save(discriminator.state_dict(), path2weights_dis)

# Load
weights = torch.load(path2weights_gen)
model_gen.load_state_dict(weights)

# evalutaion mode
model_gen.eval()

# fake image 
with torch.no_grad():
    fixed_noise = torch.randn(16, 100, device=device)
    label = torch.randint(0,7,(16,), device=device)   # Category 7#################
    img_fake = model_gen(fixed_noise, label).detach().cpu() # plot 하려면 cpu 이용
print(img_fake.shape)
#%% Example -> Create labels to fit into Coditional GAN
def create(inp):  # 7 types of Logo
  feature_map = { "lettermarks":0,
                 "Wordmarks":1,
                 "Pictorial":2,
                 "Abstract":3,
                 "Combination":4,
                 "Mascot":5,
                 "Emblem":6,
                            }
  #samples = 16
  #z_noise = np.random.uniform(-1.,1.,size = [samples, z_noise_dim])
  #one hot encoding
  Y_label = np.zeros(shape = [samples, Y_dimension])
  Y_label[:, feature_map[inp]] = 1
  
  # run the traineg generator excluding Discriminator
  generated_samples = sess.run(output_Gen, feed_dict = {z_input:z_noise, Y_input:Y_label})
  #plot images
  
  generate_plot(generated_samples)

#%%  Plot generated image

def generate_plot(samples):
  fig = plt.figure(figsize = (4,4))
  gs = gridspec.GridSpec(4,4)
  gs.update(wspace = 0.05, hspace = 0.05)
  
  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28,28), cmap = 'gray')
  return fig 

#%%
from IPython.display import Image

Image('.png')