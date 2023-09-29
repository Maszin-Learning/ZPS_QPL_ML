import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid
#from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils
from matplotlib.pyplot import figure

import spectral_analysis as sa
# DataLoader 
import os
from os import listdir
from os.path import isfile, join

device = torch.device('cpu')
dtype = torch.float32

# HyperParameters
batch_size = 1024
learning_rate = 0.005
betas=(0.5, 0.999) #beta1, beta2 for Adam Optimizer
momentum = 0.9 # for SGD optimizer




class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input 1824
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 912
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 456
            nn.Conv1d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 228
            nn.Conv1d(256, 512, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 114
            nn.Conv1d(512, 1, kernel_size=114, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = self.main(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(nz, 512, 114, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x)
        return x

    # custom weights initialization called on gen and disc model
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# set the device we will be using
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True #let cudnn choose most efficient way of calculating convulsions

generator = Generator().to(device)
generator.apply(weights_init)
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
adversarial_loss = nn.BCELoss() #change

def generator_loss(fake_output, label):
    gen_loss = adversarial_loss(fake_output, label)
    return gen_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss


G_optimizer = optim.Adam(generator.parameters(), lr = learning_rate, betas=betas)
D_optimizer = optim.SGD(discriminator.parameters(), lr = learning_rate, momentum=momentum)

def save_model(time, PATHD, PATHG):
    torch.save(discriminator.state_dict(), PATHD)
    torch.save(generator.state_dict(), PATHG)
    
def test(epoch):
    with torch.no_grad():
        generator.eval()
        noise_vector = torch.randn(16, 100, 1, 1, device=device)
        generated_image = generator(noise_vector)
        #generated_img_array = np.transpose(vutils.make_grid(generated_image, padding=2, normalize=True).cpu().detach().numpy(),(1,2,0))
        #print(generated_img_array.shape)
        #images = wandb.Image(generated_img_array, caption=f"Top: epoch {epoch}, Bottom: X")
        #wandb.log({"examples": images})
    generator.train()
    

# datetime object containing current date and time
from datetime import datetime
now = datetime.now()
print("now =", now)
time = now.strftime("%d_%m_%Y__%H_%M_%S")
PATHD = f'/notebooks/ML_logs/DGAN_D_{time}.pt'
PATHG = f'/notebooks/ML_logs/DGAN_G_{time}.pt'
torch.autograd.set_detect_anomaly(True)



#Data preparation
pulse_1 = sa.gaussian_pulse((1550,1556), 1553, 1, x_type='freq')
pulse_1.x_type = "wl"
pulse_1.wl_to_freq()
pulse_1.Y = pulse_1.Y*100
signal_len=len(pulse_1)
sa.plot(pulse_1, title = 'przed_1', save = True)

pulse_2 = sa.hermitian_pulse((1550,1556), 1553, 1, x_type='freq')
pulse_2.x_type = "wl"
pulse_2.wl_to_freq()
pulse_2.Y = pulse_2.Y*100

pulse_2_Y_abs_tensor = torch.tensor(np.abs(pulse_2.Y), requires_grad=True, device=device, dtype=dtype).reshape(1,signal_len)

train_list = [pulse_2_Y_abs_tensor]

G_loss_best = 0
D_total_loss_best = 0
generator.train()
discriminator.train()
num_epochs = 300
for epoch in tqdm(range(1, num_epochs+1)):
    D_loss_list, G_loss_list = [], []
    for index, real_images in enumerate(train_list):
        D_optimizer.zero_grad()
        real_images = real_images.to(device)
        real_target = torch.ones(real_images.shape, requires_grad=True).to(device)
        fake_target = torch.zeros(real_images.shape, requires_grad=True).to(device)
        output = discriminator(real_images)
        D_real_loss = discriminator_loss(output, real_target)
        D_real_loss.backward()

        noise_vector = torch.randn(real_images.shape, 100, 1, 1, device=device)
        noise_vector = noise_vector.to(device)
        generated_image = generator(noise_vector)
        output = discriminator(generated_image.detach())
        D_fake_loss = discriminator_loss(output,fake_target)

        # train with fake
        D_fake_loss.backward()

        D_total_loss = D_real_loss + D_fake_loss
        #D_loss_list.append(D_total_loss)

        D_optimizer.step()

        # Train G on D's output
        G_optimizer.zero_grad()
        gen_output = discriminator(generated_image)
        G_loss = generator_loss(gen_output, real_target)
        #G_loss_list.append(G_loss)

        G_loss.backward()
        G_optimizer.step()
        
        
    #print(D_total_loss.tolist(), G_loss.tolist())
    
    #
    G_loss_d = G_loss.detach()
    D_real_loss_d = D_real_loss.detach()
    D_fake_loss_d = D_fake_loss.detach()
    D_total_loss_d = D_total_loss.detach()
    
    # log train_acc and test_acc to wandb
    #wandb.log({"G_loss": G_loss_d})
    #wandb.log({"D_real_loss": D_real_loss_d})
    #wandb.log({'D_fake_loss': D_fake_loss_d})
    #wandb.log({'D_total_loss':D_total_loss_d})
    
    if epoch % 2 == 0: #generate test batch every 3 epochs
        test(epoch)
    '''    
    if G_loss_d < G_loss_best or D_total_loss_d> D_total_loss_best:
        save_model(time, PATHD, PATHG)
        G_loss_best = G_loss_d 
        D_total_loss_best = D_total_loss_d
    '''
    
discriminator.load_state_dict(torch.load(PATHD))
generator.load_state_dict(torch.load(PATHG))