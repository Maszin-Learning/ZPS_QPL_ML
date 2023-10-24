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

# phase generator to evolve our initial pulse to input pulse

def phase_gen(num, max_order = 10, max_value = None):

    if np.random.uniform(low = 0, high = 1) < 1.0:      # slowly varying phase
        X = np.linspace(-1, 1, num)
        Y = np.zeros(num)
        
        for order in range(max_order):
            coef = np.random.uniform(low = -1, high = 1)
            Y += coef*X**order
    else:                                               # rapidly varying phase  UPDATE: It causes convergence
        Y = np.zeros(num)
        for order in range(4):
            coef = np.random.uniform(low = -1, high = 1)
            Y += coef*sa.hermitian_pulse(pol_num = order,
                bandwidth = [-1, 1],
                centre = 0,
                FWHM = 0.5,
                num = num).Y
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

    tens = torch.tensor([[tens[i], 0] for i in range(len(tens))], requires_grad = True, device = my_device, dtype = my_dtype)
    tens = torch.view_as_complex(tens)
    tens = tens.reshape(1, tens.numel())

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

def pulse_gen(max_phase_value = None):

    intensity = Y_initial.copy()
    intensity = np_to_complex_pt(intensity)

    phase_significant = phase_gen(num = output_dim, 
                            max_value = np.random.uniform(low = 0, high = max_phase_value))
    phase_significant = torch.tensor(phase_significant, requires_grad = True, device = my_device, dtype = my_dtype)

    intensity = evolve(intensity, phase_significant)

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
    
    test_pulse = np_to_complex_pt(test_pulse.Y)
    test_phase = None

elif test_pulse_type == "chirp":
    test_pulse = initial_pulse.copy()
    chirp = 20
    test_phase = np_to_complex_pt(chirp*np.linspace(-1, 1, output_dim)**2)
    test_phase = test_phase.reshape([output_dim])
    test_pulse = evolve(np_to_complex_pt(test_pulse.Y), test_phase)

elif test_pulse_type == "random_evolution":
    max_phase = 20
    test_pulse, test_phase = pulse_gen(max_phase)

# ok, let's define the NN

class network(nn.Module):
    def __init__(self, input_size, n, output_size):
        super(network, self).__init__()

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

        test_phase_pred = model(test_pulse.abs())
        test_phase_pred = test_phase_pred.reshape([output_dim])

        # evolve

        initial_intensity = np_to_complex_pt(Y_initial.copy())

        test_intensity = evolve(initial_intensity, test_phase_pred)
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
                            np.abs(np.reshape(test_pulse.clone().cpu().detach().numpy(), input_dim))[plot_from:plot_to], 
                            color = "darkviolet", 
                            alpha = 0.2,
                            zorder = 0)
        plt.xlabel("THz")
        plt.legend(["Reconstructed intensity", "Initial intensity", "Target intensity"], bbox_to_anchor = [0.95, -0.15])
        plt.grid()

        plt.subplot(1, 2, 2)

        plt.title("The phase")

        reconstructed_phase = np.unwrap(test_phase_pred.clone().cpu().detach().numpy().reshape(output_dim))
        reconstructed_phase -= reconstructed_phase[floor(output_dim/2)]

        idx_start = floor(input_dim/2 - output_dim/2)
        idx_end = floor(input_dim/2 + output_dim/2)

        plt.scatter(FT_pulse.X[idx_start: idx_end], 
                    reconstructed_phase, 
                    s = 1, 
                    color = "red",
                    zorder = 10)

        if test_phase != None:
            test_phase_np = test_phase.clone().cpu().detach().numpy()
            test_phase_np -= test_phase_np[floor(output_dim/2)]
            plt.plot(FT_pulse.X[idx_start: idx_end],
                        np.real(test_phase_np),
                        color = "black",
                        lw = 1,
                        linestyle = "dashed",
                        zorder = 5)

        FT_pulse.Y /= np.max(FT_pulse.Y[idx_start: idx_end])
        if test_phase != None:
            FT_pulse.Y *= np.max(np.concatenate([np.abs(reconstructed_phase), np.abs(test_phase.clone().detach().cpu().numpy())]))
        else:
            FT_pulse.Y *= np.max(np.abs(reconstructed_phase))

        plt.fill_between(FT_pulse.X[idx_start: idx_end], 
                            np.abs(FT_pulse.Y[idx_start: idx_end]), 
                            alpha = 0.2, 
                            color = "orange",
                            zorder = 0)
        
        plt.xlabel("Quasi-time (ps)")
        if test_phase != None:
            plt.legend(["Reconstructed phase", "Initial phase", "FT intensity"], bbox_to_anchor = [0.95, -0.15])
        else:
            plt.legend(["Reconstructed phase", "FT intensity"], bbox_to_anchor = [0.95, -0.15])
        plt.grid()
        plt.savefig("pics/reconstructed{}.jpg".format(iter), bbox_inches = "tight", dpi = 200)
        plt.close()
