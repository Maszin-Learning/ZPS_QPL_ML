import torch
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
    
def evolve_np(intensity, phase, dtype, abs = True):
    '''
    ## Evolve the intensity with a sequence of operators [iFT, exp(i*phase), FT]. Suitable for NumPy.
    # Arguments:

    intensity - one-dimensional NumPy array with intensity to be transformed.

    phase - one-dimensional NumPy array with significant part of phase multiplying Fourier-transformed intensity.

    abs - if to return the absolute value of the evolved intensity.
    
    # Returns

    One-dimensional NumPy array with evolved intensity.

    # Note:
    If the phase is shorter than the intensity, only the middle part of FT(intensity) is multiplied by exp(i*phase),
    outside that area phase is assumed to be zero.
    '''
    input_dim = intensity.shape[-1]
    output_dim = phase.shape[-1]

    intensity = np.fft.fftshift(intensity)
    intensity = np.fft.fft(intensity)
    intensity = np.fft.fftshift(intensity)
    
    zeroes_shape = np.array(phase.shape)
    zeroes_shape[-1] = floor((input_dim-output_dim)/2)
    zeroes_shape = tuple(zeroes_shape)
    long_phase = np.concatenate([np.zeros(shape = zeroes_shape, dtype = dtype), 
                          phase,
                          np.zeros(shape = zeroes_shape, dtype = dtype)], axis=phase.ndim-1)
    
    complex_intensity = intensity*np.exp(1j*long_phase)

    complex_intensity = np.fft.ifftshift(complex_intensity)
    complex_intensity = np.fft.ifft(complex_intensity)
    complex_intensity = np.fft.ifftshift(complex_intensity)

    if abs:
        return np.abs(complex_intensity)
    else:
        return complex_intensity
    

def evolve_pt(intensity, phase, device, dtype, abs = True):
    '''
    ## Evolve the intensity with a sequence of operators [iFT, exp(i*phase), FT]. Suitable for PyTorch. Works with batches.
    # Arguments:

    intensity - PyTorch Tensor with intensity to be transformed. Intensity's shape = [1, N].

    phase - one-dimensional NumPy array with significant part of phase multiplying Fourier-transformed intensity.

    abs - if to return the absolute value of the evolved intensity.
    
    # Returns

    One-dimensional NumPy array with evolved intensity.

    # Note:
    If the phase is shorter than the intensity, only the middle part of FT(intensity) is multiplied by exp(i*phase),
    outside that area phase is assumed to be zero.
    '''

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
    
    
def plot_dataset(number, pulse, ft_pulse):
    '''
    # Plot \"number\" of phases and intensities that are saved as .csv files in \"data\" folder.
    \"pulse\" is a spectrum class object representing the initial - not transformed - pulse. 
    \"ft_pulse\" is a spectrum class object of the same length as the phase in the dataset. It is the Fourier transform of
    the initial pulse.
    '''

    # prepare place for saving plots

    if os.path.isdir("dataset_sample"):
        shutil.rmtree('dataset_sample')
    os.mkdir("dataset_sample")

    # check how the dataset looks like

    intensity_labels = os.listdir('data/train_intensity')
    phase_labels = os.listdir('data/train_intensity')
    dataset_size = len(intensity_labels)

    if len(intensity_labels) != len(phase_labels):
        raise Exception("Number of generated phases is not equal to number of the intensities.")
    if len(intensity_labels) < number:
        raise Exception("That's not very good idea to save more plots that the amount of data you have.")
    
    # pick random examples to plot

    plot_indices = np.random.randint(low = 0, high = dataset_size, size = number)

    # and plot them

    pulse_safe = ft_pulse.copy()

    print("Saving example phases and intensities...")

    for n in tqdm(range(len(plot_indices))):
        i = plot_indices[n]

        phase = np.loadtxt("data/train_phase/" + phase_labels[i])
        intensity = np.loadtxt("data/train_intensity/" + intensity_labels[i])

        pulse_safe.Y /= np.max(np.abs(pulse_safe.Y))
        pulse_safe.Y *= np.max(np.abs(phase))

        plt.subplot(2, 1, 2)
        plt.fill_between(pulse_safe.X, pulse_safe.Y, color = "orange", alpha = 0.4)
        plt.scatter(pulse_safe.X, phase, color = "red", s = 9)
        plt.grid()
        plt.legend(["Spectral intensity", "Spectral phase"])
        plt.title("Train phase")
        plt.xlabel("Quasitime (ps)")
        plt.ylabel("Spectral phase (rad)")

        plt.subplot(2, 1, 1)
        plt.plot(pulse.X, intensity, color = "darkorange")
        plt.plot(pulse.X, pulse.Y, color = "black", linestyle = "dashed")
        plt.grid()
        plt.legend(["New intensity", "Initial intensity"])
        plt.title("Train intensity")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Intensity (a.u.)")

        plt.tight_layout()
        plt.savefig("dataset_sample/{}.jpg".format(i + 1))
        plt.close()

    print("Saving completed.\n")

def np_to_complex_pt(array, device, dtype):
    '''
    Transform one-dimensional real-valued NumPy array into complex-valued PyTorch Tensor with shape [1, len(array)].
    '''
    array = torch.tensor([[array[i], 0] for i in range(len(array))], requires_grad = True, device = device, dtype = dtype)
    array = torch.view_as_complex(array)
    array = array.reshape(1, array.numel())

    return array


def TB_prod(time, freq, time_intensity, freq_intensity):
    
    def sigma(X, Y):
        the_mean = np.sum(Y*X)/np.sum(np.abs(Y))
        return np.sqrt(np.sum((X-the_mean)**2 * Y)/np.sum(Y))##to zwraca poprawne wyniki bez dx
    
    std_time = sigma(time, np.abs(time_intensity)**2)
    print(std_time*2.355*1000, "time")

    std_freq = sigma(freq, np.abs(freq_intensity)**2)
    print(std_freq*2.355, "freq")

    TBp = std_freq*std_time*(2.355**2)
    print("Time bandwidth product: {}".format(TBp))

    return TBp

def freq_to_time(freq_intensity):
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(freq_intensity)))


def time_to_freq(time_intensity):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(time_intensity)))