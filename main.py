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

my_dtype = torch.float32

# "target" pulse (to be reconstructed later on)

input_dim = 1000 # number of points in single pulse

target_pulse = sa.hermitian_pulse(pol_num = 0,
                                  bandwidth = [190, 196],
                                  centre = 193,
                                  FWHM = 0.5,
                                  num = input_dim)

Y_target = target_pulse.Y.copy()

# we want to find what is the bandwidth of intensity after FT, to estimate output dimension of NN

target_pulse_2 = target_pulse.copy()
target_pulse_2.fourier()
x_start = target_pulse_2.quantile(0.001)
x_end = target_pulse_2.quantile(0.999)
idx_start = np.searchsorted(target_pulse_2.X, x_start)
idx_end = np.searchsorted(target_pulse_2.X, x_end)
output_dim = idx_end - idx_start    # number of points of non-zero FT-intensity
if output_dim % 2 == 1:
    output_dim += 1

# phase generator to evolve our target pulse to, well... input pulse

def phase_gen(num, max_order = 10, max_value = None, chirp = None):
    X = np.linspace(-1, 1, num)
    Y = np.zeros(num)
    for order in range(max_order):
        coef = np.random.uniform(low = -1, high = 1)
        Y += coef*X**order
    if chirp != None:
        return chirp*X**2
    else:
        if max_value == None:
            return Y
        else:
            return Y/np.max(np.abs(Y))*max_value

you_dont_trust_me_that_these_phases_look_cool = False

if you_dont_trust_me_that_these_phases_look_cool:
    for i in range(10):
        phase = phase_gen(100, 10)
        plt.plot(np.linspace(0, 1, 100), phase, color = "deeppink")
        plt.grid()
        plt.title("Test phase")
        plt.savefig("phase_{}.jpg".format(i + 1))
        plt.close()

# now function to provide input to the network

def pulse_gen(max_phase_value = None, chirp = None):

    intensity = Y_target.copy()

    intensity = torch.tensor([[intensity[i], 0] for i in range(input_dim)], requires_grad = True, device = my_device, dtype = my_dtype)
    intensity = torch.view_as_complex(intensity)
    intensity = intensity.reshape(1, input_dim)

    intensity = torch.fft.fftshift(intensity)
    intensity = torch.fft.fft(intensity)
    intensity = torch.fft.fftshift(intensity)

    phase_signif = phase_gen(num = output_dim, 
                            max_value = max_phase_value, 
                            chirp = chirp)
    phase_signif = torch.tensor(phase_signif, requires_grad = True, device = my_device, dtype = my_dtype)

    phase = torch.concat([torch.zeros(size = [floor((input_dim-output_dim)/2)], requires_grad = True, device = my_device, dtype = my_dtype), 
                          phase_signif,
                          torch.zeros(size = [floor((input_dim-output_dim)/2)], requires_grad = True, device = my_device, dtype = my_dtype)])
    
    complex_intensity = torch.mul(intensity, torch.exp(1j*phase))

    complex_intensity = torch.fft.ifftshift(complex_intensity)
    complex_intensity = torch.fft.ifft(complex_intensity)
    complex_intensity = torch.fft.ifftshift(complex_intensity)

    return complex_intensity.abs(), phase_signif

# ok, let's define the NN

class network(nn.Module):
    def __init__(self, input_size, n, output_size):
        super(network, self).__init__()

        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)9
        self.linear_3 = nn.Linear(n,output_size)
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)

    def forward(self,x):
        x = self.leakyrelu(self.linear_1(x))
        x = self.normal_1(x)
        #x = self.leakyrelu(self.linear_2(x))
        x = self.linear_3(x)
        x = self.normal_3(x)
        return self.leakyrelu(x)
    
# create NN

model = network(input_size = input_dim, 
                n = 100, 
                output_size = output_dim)
model.to(device = my_device, dtype = my_dtype)
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = torch.nn.MSELoss()

# test pulse (just chirp)

test_pulse, test_phase = pulse_gen(chirp = 1)

# training loop

iteration_num = 10000

loss_list = []

for iter in tqdm(range(iteration_num)):

    # generate the pulse for this iteration

    pulse, phase = pulse_gen()

    # predict phase that will transform gauss into this pulse

    predicted_phase = model(pulse)
    predicted_phase = predicted_phase.reshape([output_dim])

    # transform gauss into something using this phase

    initial_intensity = torch.tensor(Y_target.copy(), requires_grad = True,device = my_device, dtype = my_dtype)
    target_intensity = initial_intensity.clone()

    initial_intensity = torch.fft.fftshift(initial_intensity)
    initial_intensity = torch.fft.fft(initial_intensity)
    initial_intensity = torch.fft.fftshift(initial_intensity)

    phase_2 = torch.concat([torch.zeros(size = [floor((input_dim-output_dim)/2)], 
                                        requires_grad = True, 
                                        device = my_device, 
                                        dtype = my_dtype), 
                        predicted_phase,
                        torch.zeros(size = [floor((input_dim-output_dim)/2)], 
                                    requires_grad = True, 
                                    device = my_device, 
                                    dtype = my_dtype)])
    
    complex_intensity = torch.mul(initial_intensity, torch.exp(1j*phase_2))

    complex_intensity = torch.fft.ifftshift(complex_intensity)
    complex_intensity = torch.fft.ifft(complex_intensity)
    complex_intensity = torch.fft.ifftshift(complex_intensity)

    # a bit of calculus
    
    loss = criterion(complex_intensity.abs(), target_intensity.abs())
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

        plot_from = 200
        plot_to = 800

        # generate test chirp pulse

        chirp_pulse, chirp_phase = pulse_gen(chirp = 1)

        # compute phase that should evolve gauss to this pulse

        chirp_phase_pred = model(chirp_pulse)
        chirp_phase_pred = chirp_phase_pred.reshape([output_dim])

        # evolve

        initial_intensity = torch.tensor(Y_target.copy(), requires_grad = True,device = my_device, dtype = my_dtype)

        initial_intensity = torch.fft.fftshift(initial_intensity)
        initial_intensity = torch.fft.fft(initial_intensity)
        initial_intensity = torch.fft.fftshift(initial_intensity)

        phase_3 = torch.concat([torch.zeros(size = [floor((input_dim-output_dim)/2)], 
                                            requires_grad = True, 
                                            device = my_device, 
                                            dtype = my_dtype), 
                            chirp_phase_pred,
                            torch.zeros(size = [floor((input_dim-output_dim)/2)], 
                                        requires_grad = True, 
                                        device = my_device, 
                                        dtype = my_dtype)])
        
        complex_intensity = torch.mul(initial_intensity, torch.exp(1j*phase_3))

        complex_intensity = torch.fft.ifftshift(complex_intensity)
        complex_intensity = torch.fft.ifft(complex_intensity)
        complex_intensity = torch.fft.ifftshift(complex_intensity)

        reconstructed = complex_intensity.abs() 

        plt.subplot(1, 2, 1)
        plt.title("The intensity")

        plt.scatter(target_pulse.X[plot_from:plot_to], 
                    np.reshape(reconstructed.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to], 
                    color = "green", 
                    s = 1,
                    zorder = 10)
        plt.plot(target_pulse.X[plot_from:plot_to], 
                    target_pulse.Y[plot_from:plot_to], 
                    linestyle = "dashed", 
                    color = "black", 
                    lw = 1,
                    zorder = 5)
        plt.fill_between(target_pulse.X[plot_from:plot_to], 
                            np.reshape(chirp_pulse.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to], 
                            color = "darkviolet", 
                            alpha = 0.2,
                            zorder = 0)
        plt.xlabel("THz")
        plt.legend(["Reconstructed intensity", "Initial intensity", "Target intensity"], bbox_to_anchor = [0.95, -0.15])
        plt.grid()

        plt.subplot(1, 2, 2)

        plt.title("The phase")
        
        phase_start = floor((input_dim - 1*output_dim)/2)
        phase_end = floor((input_dim + 1*output_dim)/2)

        phase_final = np.unwrap(phase_3.clone().cpu().detach().numpy().reshape(input_dim))
        phase_final -= round(phase_final[floor(input_dim/2)])
        plt.scatter(range(phase_end - phase_start), 
                    phase_final[phase_start: phase_end], 
                    s = 1, 
                    color = "red",
                    zorder = 10)
        plt.plot(range(phase_end - phase_start),
                    chirp_phase.clone().cpu().detach().numpy(),
                    color = "black",
                    lw = 1,
                    linestyle = "dashed",
                    zorder = 5)
        '''
        ft_intensity /= np.max(ft_intensity[phase_start: phase_end])
        ft_intensity *= np.max(np.concatenate([phase_final[phase_start: phase_end], phase_init[phase_start: phase_end]]))
        plt.fill_between(phase_X[phase_start: phase_end], 
                            np.abs(ft_intensity[phase_start: phase_end]), 
                            alpha = 0.2, 
                            color = "orange",
                            zorder = 0)
        '''
        plt.xlabel("Quasi-time (ps)")
        plt.legend(["Reconstructed phase", "Initial phase", "FT intensity"], bbox_to_anchor = [0.95, -0.15])
        plt.grid()
        plt.savefig("pics/reconstructed{}.jpg".format(iter), bbox_inches = "tight", dpi = 200)
        plt.close()
