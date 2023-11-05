# modules

import spectral_analysis as sa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from math import floor
from utilities import evolve, np_to_complex_pt, plot_phases
from test import test
from dataset import Dataset
from torch.utils.data import DataLoader #Dataloader module
from test import create_test_pulse

# cuda 

if torch.cuda.is_available():
    my_device = torch.device("cuda")
    print (f"Using {my_device}")
    print('allocated CUDA memory: ',torch.cuda.memory_allocated())      # Checking GPU RAM allocated memory
    print('cached CUDA memory: ',torch.cuda.memory_cached())
    torch.cuda.empty_cache()                                            # clear CUDA memory
    torch.backends.cudnn.benchmark = True                               # let cuda chose the most efficient way of calculating Convolutions
    
elif torch.backends.mps.is_available():
    print ("CUDA device not found.")
    my_device = torch.device("mps")
    print (f"Using {my_device}")
else:
    print ("MPS device not found.")
    my_device = torch.device("cpu")
    print (f"Using {my_device}")

# data type

my_device = torch.device('cpu')
my_dtype = torch.float32

# initial pulse (to be reconstructed later on)

input_dim = 2000 # number of points in single pulse

bandwidth = [190, 196]
centre = [193]
FWHM = 0.25

initial_pulse = sa.hermitian_pulse(pol_num = 0,
                                  bandwidth = bandwidth,
                                  centre = centre,
                                  FWHM = FWHM,
                                  num = input_dim)

Y_initial = initial_pulse.Y.copy()

FT_pulse = initial_pulse.fourier(inplace = False)

# we want to find what is the bandwidth of intensity after FT, to estimate output dimension of NN

initial_pulse_2 = initial_pulse.copy()
initial_pulse_2.fourier()
x_start = initial_pulse_2.quantile(0.001)
x_end = initial_pulse_2.quantile(0.999)
reconstructed_phase = np.searchsorted(initial_pulse_2.X, x_start)
idx_end = np.searchsorted(initial_pulse_2.X, x_end)
output_dim = idx_end - reconstructed_phase    # number of points of non-zero FT-intensity
if output_dim % 2 == 1:
    output_dim += 1

print("\ninput_dim (spectrum length) = {}".format(input_dim))  
print("output_dim (phase length) = {}".format(output_dim))

# test pulse

test_pulse, test_phase = create_test_pulse("hermite", initial_pulse, output_dim, my_device, my_dtype)

# ok, let's define the NN

class network(nn.Module):
    def __init__(self, input_size, n, output_size):
        super(network, self).__init__()
        self.input = input_size
        self.output = output_size

        self.linear_1 = nn.Linear(input_size, n)
        self.linear_2 = nn.Linear(n, n)
        self.linear_3 = nn.Linear(n, output_size)
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace = True)
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        x = self.leakyrelu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_3(x)
        return x
    
# create NN

model = network(input_size = input_dim, 
                n = 100, 
                output_size = output_dim)
model.to(device = my_device, dtype = my_dtype)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = torch.nn.MSELoss()

# training loop

iteration_num = 20
_batch_size = 32
batch_num = 500
loss_list = []

# create dataset and wrap it into dataloader

print("\nCreating training set...")

dataset_train = Dataset(batch_num = batch_num,
                        batch_size = _batch_size,
                        initial_intensity = Y_initial,
                        phase_len = output_dim, 
                        device = my_device, 
                        dtype = my_dtype, 
                        max_order = 10, 
                        max_value = None)

print("Training set created. It contains {} examples grouped into {}-element long batches.\n".format(_batch_size*batch_num, _batch_size))

dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=_batch_size, num_workers=0)

for epoch in tqdm(range(iteration_num)):

    dataset_train = Dataset(batch_num = batch_num,
                        batch_size = _batch_size,
                        initial_intensity = Y_initial,
                        phase_len = output_dim, 
                        device = my_device, 
                        dtype = my_dtype, 
                        max_order = 10, 
                        max_value = None)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=_batch_size, num_workers=0)

    for pulse, _ in dataloader_train:

        # predict phase that will transform gauss into this pulse

        predicted_phase = model(pulse)
        predicted_phase = predicted_phase

        # transform gauss into something using this phase

        initial_intensity = np_to_complex_pt(Y_initial.copy(), device = my_device, dtype = my_dtype)
        reconstructed_intensity = evolve(initial_intensity, predicted_phase, device = my_device, dtype = my_dtype)

        # a bit of calculus

        loss = criterion(reconstructed_intensity.abs(), pulse) # pulse intensity
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # stats

        loss_list.append(loss.clone().cpu().detach().numpy())

    if True:
        #if epoch == 0:
            #print("Iteration np. {}. Loss {}.".format(epoch, loss.clone().cpu().detach().numpy()))
        print("Epoch no. {}. Loss {}.".format(epoch, np.mean(np.array(loss_list[epoch*len(dataloader_train): (epoch+1)*len(dataloader_train)]))))

        test(model = model,
                test_pulse = test_pulse,
                test_phase = test_phase,
                initial_pulse_Y = initial_pulse.Y.copy(),
                initial_pulse_X = initial_pulse.X.copy(),
                device = my_device, 
                dtype = my_dtype,
                iter_num =epoch)
