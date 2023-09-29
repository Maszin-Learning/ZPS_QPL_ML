import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils
from matplotlib.pyplot import figure

# DataLoader 
import os
from os import listdir
from os.path import isfile, join



# HyperParameters
batch_size = 1024
learning_rate = 0.005
betas=(0.5, 0.999) #beta1, beta2 for Adam Optimizer
momentum = 0.9 # for SGD optimizer

# Generator Model Class Definition with dropouts
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Block 1:input is Z, going into a convolution
            nn.ConvTranspose1d(100, 64 * 8, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.5,inplace=True),
            # Block 2: input is (64 * 8) x 4 x 4
            nn.ConvTranspose1d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.5,inplace=True),
            # Block 3: input is (64 * 4) x 8 x 8
            nn.ConvTranspose1d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.5,inplace=True),
            # Block 4: input is (64 * 2) x 16 x 16
            nn.ConvTranspose1d(64 * 2, 64, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.5,inplace=True),
            # Block 5: input is (64) x 32 x 32
            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: output is (3) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

# Discriminator Model Class Definition
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Block 1: input is (3) x 64 x 64
            nn.Conv1d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.5,inplace=True),
            # Block 2: input is (64) x 32 x 32
            nn.Conv1d(64, 64 * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.5,inplace=True),
            # Block 3: input is (64*2) x 16 x 16
            nn.Conv1d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.5,inplace=True),
            # Block 4: input is (64*4) x 8 x 8
            nn.Conv1d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.5,inplace=True),
            # Block 5: input is (64*8) x 4 x 4
            nn.Conv1d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.Flatten()
            # Output: 1
        )

    def forward(self, input):
        output = self.main(input)
        output  = output.reshape(output.shape[0])
        return output

    # custom weights initialization called on gen and disc model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
        
# set the device we will be using
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


G_loss_best = 0
D_total_loss_best = 0
generator.train()
discriminator.train()
num_epochs = 300
for epoch in tqdm(range(1, num_epochs+1)):
    D_loss_list, G_loss_list = [], []
    for index, (real_images, _) in enumerate(train_loader):
        D_optimizer.zero_grad()
        real_images = real_images.to(device)
        real_target = torch.ones(real_images.size(0), requires_grad=True).to(device)
        fake_target = torch.zeros(real_images.size(0), requires_grad=True).to(device)
        output = discriminator(real_images)
        D_real_loss = discriminator_loss(output, real_target)
        D_real_loss.backward()

        noise_vector = torch.randn(real_images.size(0), 100, 1, 1, device=device)
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