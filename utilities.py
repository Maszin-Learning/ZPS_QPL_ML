import torch
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import os
import spectral_analysis as sa
    
def evolve_np(intensity, phase, dtype, abs = True):

    if dtype == torch.float32:
        new_dtype = np.float32
    else:
        new_dtype = dtype

    input_dim = intensity.shape[-1]
    output_dim = phase.shape[-1]

    intensity = np.fft.fftshift(intensity)
    intensity = np.fft.fft(intensity)
    intensity = np.fft.fftshift(intensity)
    
    zeroes_shape = np.array(phase.shape)
    zeroes_shape[-1] = floor((input_dim-output_dim)/2)
    zeroes_shape = tuple(zeroes_shape)
    long_phase = np.concatenate([np.zeros(shape = zeroes_shape, dtype = new_dtype), 
                          phase,
                          np.zeros(shape = zeroes_shape, dtype = new_dtype)], axis=phase.ndim-1)
    
    complex_intensity = intensity*np.exp(1j*long_phase)

    complex_intensity = np.fft.ifftshift(complex_intensity)
    complex_intensity = np.fft.ifft(complex_intensity)
    complex_intensity = np.fft.ifftshift(complex_intensity)

    if abs:
        return np.abs(complex_intensity)
    else:
        return complex_intensity
    

def evolve_pt(intensity, phase, device, dtype, abs = True):

    input_dim = intensity.shape[-1]
    output_dim = phase.shape[-1]

    intensity = torch.fft.fftshift(intensity)
    intensity = torch.fft.fft(intensity)
    intensity = torch.fft.fftshift(intensity)
    
    zeroes_shape = np.array(phase.shape)
    zeroes_shape[-1] = floor((input_dim-output_dim)/2)
    zeroes_shape = tuple(zeroes_shape)
    long_phase = torch.concat([torch.zeros(size = zeroes_shape, requires_grad = True, device = device, dtype = dtype), 
                          phase,
                          torch.zeros(size = zeroes_shape, requires_grad = True, device = device, dtype = dtype)], dim=phase.ndim-1)
    
    complex_intensity = torch.mul(intensity, torch.exp(1j*long_phase))

    complex_intensity = torch.fft.ifftshift(complex_intensity)
    complex_intensity = torch.fft.ifft(complex_intensity)
    complex_intensity = torch.fft.ifftshift(complex_intensity)

    if abs:
        return complex_intensity.abs()
    else:
        return complex_intensity
    
    
def plot_phases(phase_generator, num, phase_type = "regular"):

    if not os.path.isdir("example_phases"):
        os.mkdir("example_phases")

    for i in range(num):
        phase = phase_generator(100, phase_type = phase_type)
        plt.plot(np.linspace(0, 1, 100), phase, color = "deeppink")
        plt.grid()
        plt.title("Test phase")
        plt.savefig("example_phases/phase_{}.jpg".format(i + 1))
        plt.close()


def np_to_complex_pt(array, device, dtype):
    '''
    # real-valued numpy array to complex pytorch tensor
    '''


    array = torch.tensor([[array[i], 0] for i in range(len(array))], requires_grad = True, device = device, dtype = dtype)
    array = torch.view_as_complex(array)
    array = array.reshape(1, array.numel())

    return array


def TB_prod(t, f, Ut, Uf):
    
    def sigma(x, a):
        mu = np.sum(a*x)/np.sum(np.abs(a))
        return np.sqrt(np.sum((x-mu)**2 * a)/np.sum(a))##to zwraca poprawne wyniki bez dx
    
    S_t = sigma(t, np.abs(Ut)**2)
    print((S_t)*2.355*1000, "time")
    S_f = sigma(f, np.abs(Uf)**2)
    print((S_f)*2.355, "freq")
    TBp = S_f*S_t*(2.355**2)
    print("Time bandwidth product: ",TBp)
    return TBp

def U_f_in__to__U_t_out(U_f_in):
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(U_f_in)))


def U_t_in__to__U_f_out(U_t_in):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(U_t_in)))