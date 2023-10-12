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

# target pulse (to be reconstructed later on)

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

def phase_gen(num, max_order = 10, max_value = None):
    X = np.linspace(-1, 1, num)
    Y = np.zeros(num)
    for order in range(max_order):
        coef = np.random.uniform(low = -1, high = 1)
        Y += coef*X**order
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

def pulse_gen(max_phase_value = None):

    intensity = Y_target.copy()

    intensity = torch.tensor([[intensity[i], 0] for i in range(input_dim)], requires_grad = True,)
    intensity = torch.view_as_complex(intensity)
    intensity = intensity.reshape(1, input_dim)

    intensity = torch.fft.fftshift(intensity)
    intensity = torch.fft.fft(intensity)
    intensity = torch.fft.fftshift(intensity)

    phase_signif = torch.tensor(phase_gen(num = output_dim, max_value = max_phase_value), requires_grad = True)
    phase = torch.concat([torch.zeros(size = [floor((input_dim-output_dim)/2)], requires_grad = True), 
                          phase_signif,
                          torch.zeros(size = [floor((input_dim-output_dim)/2)], requires_grad = True)])
    
    complex_intensity = torch.mul(intensity, torch.exp(1j*phase))

    complex_intensity = torch.fft.ifftshift(complex_intensity)
    complex_intensity = torch.fft.ifft(complex_intensity)
    complex_intensity = torch.fft.ifftshift(complex_intensity)

    return complex_intensity.abs()

for i in range(10):

    pulse = pulse_gen(max_phase_value = 20)
    plt.plot(target_pulse.X, np.reshape(pulse.detach().numpy(), 1000), color = "deeppink", lw = 1)
    plt.plot(target_pulse.X, target_pulse.Y, color = "blue", lw = 1)
    plt.savefig("pulse_{}.jpg".format(i + 1))
    plt.close()