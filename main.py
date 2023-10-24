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
from dataset_generator import Generator as Gen
from utilities import evolve, np_to_complex_pt
from test import test

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

print("input_dim (spectrum length) = {}".format(input_dim))  
print("output_dim (phase length) = {}".format(output_dim))


phase_generator_1 = Gen(100,10)


you_dont_trust_me_that_these_phases_look_cool = True

if you_dont_trust_me_that_these_phases_look_cool:
    for i in range(10):
        phase = phase_generator_1.phase_gen()
        plt.plot(np.linspace(0, 1, 100), phase, color = "deeppink")
        plt.grid()
        plt.title("Test phase")
        plt.savefig("phase_{}.jpg".format(i + 1))
        plt.close()


    
# now function to provide input to the network

def pulse_gen(max_phase_value = None, phase_type = "regular"):


    intensity = Y_initial.copy()
    intensity = np_to_complex_pt(intensity, device = my_device, dtype = my_dtype)
    
    phase_generator_2 = Gen(num = output_dim, 
                            max_value = np.random.uniform(low = 0, high = max_phase_value))
    phase_significant = phase_generator_2.phase_gen()

    phase_significant = torch.tensor(phase_significant, requires_grad = True, device = my_device, dtype = my_dtype)
    
    intensity = evolve(intensity, phase_significant, device = my_device, dtype = my_dtype)

    del phase_generator_2
    
    return intensity.abs(), phase_significant

# test pulse

test_pulse_type = "hermite"

if test_pulse_type == "hermite":
    test_pulse = sa.hermitian_pulse(pol_num = 1,
                                    bandwidth = bandwidth,
                                    centre = 193,
                                    FWHM = 1,
                                    num = input_dim)

    test_pulse.Y /= np.sum(test_pulse.Y*np.conjugate(test_pulse.Y))
    test_pulse.Y *= np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y))
    
    test_pulse = np_to_complex_pt(test_pulse.Y, device = my_device, dtype = my_dtype)
    test_phase = None

elif test_pulse_type == "chirp":
    test_pulse = initial_pulse.copy()
    chirp = 20
    test_phase = np_to_complex_pt(chirp*np.linspace(-1, 1, output_dim)**2, device = my_device, dtype = my_dtype)
    test_phase = test_phase.reshape([output_dim])
    test_pulse = evolve(np_to_complex_pt(test_pulse.Y, device = my_device, dtype = my_dtype), test_phase, device = my_device, dtype = my_dtype)

elif test_pulse_type == "random_evolution":
    max_phase = 20
    test_pulse, test_phase = pulse_gen(max_phase)

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

iteration_num = 20000

loss_list = []

for iter in tqdm(range(iteration_num)):

    # generate the pulse for this iteration

    pulse, phase = pulse_gen(max_phase_value = 25)

    # predict phase that will transform gauss into this pulse

    predicted_phase = model(pulse)
    predicted_phase = predicted_phase.reshape([output_dim])

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

    stat_time = 500
    if iter % stat_time == 0:
        if iter == 0:
            print("Iteration np. {}. Loss {}.".format(iter, loss.clone().cpu().detach().numpy()))
        else:
            print("Iteration np. {}. Loss {}.".format(iter, np.mean(np.array(loss_list[iter-stat_time: iter]))))

            test(model = model,
                 test_pulse = test_pulse,
                 initial_pulse_Y = initial_pulse.Y.copy(),
                 initial_pulse_X = initial_pulse.X.copy(),
                 device = my_device, 
                 dtype = my_dtype,
                 iter_num = iter)