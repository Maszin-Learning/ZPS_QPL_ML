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

# initial pulse (to be reconstructed later on)

input_dim = 2000 # number of points in single pulse

initial_pulse = sa.hermitian_pulse(pol_num = 0,
                                  bandwidth = [190, 196],
                                  centre = 193,
                                  FWHM = 0.25,
                                  num = input_dim)

Y_initial = initial_pulse.Y.copy()

# hermit pulse

hermitian_pulse = sa.hermitian_pulse(pol_num = 1,
                                  bandwidth = [190, 196],
                                  centre = 193,
                                  FWHM = 1,
                                  num = input_dim)

hermitian_pulse.Y /= np.sum((hermitian_pulse.Y)*np.conjugate(hermitian_pulse.Y))
hermitian_pulse.Y *= np.sum((initial_pulse.Y)*np.conjugate(initial_pulse.Y))

# we want to find what is the bandwidth of intensity after FT, to estimate output dimension of NN

initial_pulse_2 = initial_pulse.copy()
initial_pulse_2.fourier()
x_start = initial_pulse_2.quantile(0.001)
x_end = initial_pulse_2.quantile(0.999)
idx_start = np.searchsorted(initial_pulse_2.X, x_start)
idx_end = np.searchsorted(initial_pulse_2.X, x_end)
output_dim = idx_end - idx_start    # number of points of non-zero FT-intensity
if output_dim % 2 == 1:
    output_dim += 1
print("output_dim = {}".format(output_dim))
print("input_dim = {}".format(input_dim))

# phase generator to evolve our initial pulse to input pulse

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

you_dont_trust_me_that_these_phases_look_cool = True

if you_dont_trust_me_that_these_phases_look_cool:
    for i in range(10):
        phase = phase_gen(100, 10)
        plt.plot(np.linspace(0, 1, 100), phase, color = "deeppink")
        plt.grid()
        plt.title("Test phase")
        plt.savefig("phase_{}.jpg".format(i + 1))
        plt.close()

# real-valued numpy array to complex pytorch tensor

def np_to_complex_pt(tens):

    tens = torch.tensor([[tens[i], 0] for i in range(input_dim)], requires_grad = True, device = my_device, dtype = my_dtype)
    tens = torch.view_as_complex(tens)
    tens = tens.reshape(1, input_dim)

    return tens

# evolution by given phase operator

def evolve(intensity, phase, abs = True):

    intensity = torch.fft.fftshift(intensity)
    intensity = torch.fft.fft(intensity)
    intensity = torch.fft.fftshift(intensity)
    
    long_phase = torch.concat([torch.zeros(size = [floor((input_dim-output_dim)/2)], requires_grad = True, device = my_device, dtype = my_dtype), 
                          phase,
                          torch.zeros(size = [floor((input_dim-output_dim)/2)], requires_grad = True, device = my_device, dtype = my_dtype)])
    
    complex_intensity = torch.mul(intensity, torch.exp(1j*long_phase))

    complex_intensity = torch.fft.ifftshift(complex_intensity)
    complex_intensity = torch.fft.ifft(complex_intensity)
    complex_intensity = torch.fft.ifftshift(complex_intensity)

    if abs:
        return complex_intensity.abs()
    else:
        return complex_intensity
    
# now function to provide input to the network

def pulse_gen(max_phase_value = None, chirp = None):

    intensity = Y_initial.copy()
    intensity = np_to_complex_pt(intensity)

    phase_significant = phase_gen(num = output_dim, 
                            max_value = max_phase_value, 
                            chirp = chirp)
    phase_significant = torch.tensor(phase_significant, requires_grad = True, device = my_device, dtype = my_dtype)

    intensity = evolve(intensity, phase_significant)

    return intensity.abs(), phase_significant

# ok, let's define the NN

class network(nn.Module):
    def __init__(self, input_size, n, output_size):
        super(network, self).__init__()

        self.linear_1 = nn.Linear(input_size, n)
        self.linear_2 = nn.Linear(n, n)
        self.linear_3 = nn.Linear(n,output_size)
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)

        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        x = self.leakyrelu(self.linear_1(x))
        #x = self.normal_1(x)
        #x = self.leakyrelu(self.linear_2(x))
        x = self.dropout(x)
        x = self.linear_3(x)
        #x = self.normal_3(x)
        return self.leakyrelu(x)
    
# create NN

model = network(input_size = input_dim, 
                n = 200, 
                output_size = output_dim)
model.to(device = my_device, dtype = my_dtype)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = torch.nn.MSELoss()

# test pulse (just chirp)

test_pulse, test_phase = pulse_gen(chirp = 1)

# training loop

iteration_num = 20000

loss_list = []

for iter in tqdm(range(iteration_num)):

    # generate the pulse for this iteration

    pulse, phase = pulse_gen(max_phase_value = 7)

    # predict phase that will transform gauss into this pulse

    predicted_phase = model(pulse)
    predicted_phase = predicted_phase.reshape([output_dim])

    # transform gauss into something using this phase

    initial_intensity = np_to_complex_pt(Y_initial.copy())

    reconstructed_intensity = evolve(initial_intensity, predicted_phase)

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

        plot_from = floor(input_dim*1/6)
        plot_to = floor(input_dim*5/6)

        # generate test chirp pulse

        chirp_pulse, chirp_phase = pulse_gen(chirp = 20)

        hermitian_Y = hermitian_pulse.Y.copy()
        chirp_pulse = np_to_complex_pt(hermitian_Y)

        #chirp_pulse = pulse
        #chirp_phase = phase

        # compute phase that should evolve gauss to this pulse

        chirp_phase_pred = model(chirp_pulse.abs())
        chirp_phase_pred = chirp_phase_pred.reshape([output_dim])

        # evolve

        initial_intensity = np_to_complex_pt(Y_initial.copy())

        test_intensity = evolve(initial_intensity, chirp_phase_pred)
        reconstructed = test_intensity.abs() 

        plt.subplot(1, 2, 1)
        plt.title("The intensity")

        plt.scatter(initial_pulse.X[plot_from:plot_to], 
                   np.abs(np.reshape(reconstructed.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to]), 
                    color = "green", 
                    s = 1,
                    zorder = 10)
        plt.plot(initial_pulse.X[plot_from:plot_to], 
                    np.abs(initial_pulse.Y[plot_from:plot_to]), 
                    linestyle = "dashed", 
                    color = "black", 
                    lw = 1,
                    zorder = 5)
        plt.fill_between(initial_pulse.X[plot_from:plot_to], 
                            np.abs(np.reshape(chirp_pulse.clone().cpu().detach().numpy(), input_dim))[plot_from:plot_to], 
                            color = "darkviolet", 
                            alpha = 0.2,
                            zorder = 0)
        plt.xlabel("THz")
        plt.legend(["Reconstructed intensity", "Initial intensity", "Target intensity"], bbox_to_anchor = [0.95, -0.15])
        plt.grid()

        plt.subplot(1, 2, 2)

        plt.title("The phase")

        phase_final = np.unwrap(chirp_phase_pred.clone().cpu().detach().numpy().reshape(output_dim))
        phase_final -= round(phase_final[floor(output_dim/2)])
        plt.scatter(range(output_dim), 
                    phase_final, 
                    s = 1, 
                    color = "red",
                    zorder = 10)
        '''
        plt.plot(range(output_dim),
                    chirp_phase.clone().cpu().detach().numpy(),
                    color = "black",
                    lw = 1,
                    linestyle = "dashed",
                    zorder = 5)
        
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
