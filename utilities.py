import torch
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import spectral_analysis as sa
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
        plt.fill_between(pulse_safe.X, np.real(pulse_safe.Y), color = "orange", alpha = 0.4)
        plt.scatter(pulse_safe.X, np.real(phase), color = "red", s = 9)
        plt.grid()
        plt.legend(["Temporal intensity", "Temporal phase"])
        plt.title("Train phase")
        plt.xlabel("Quasitime (ps)")
        plt.ylabel("Temporal phase (rad)")

        plt.subplot(2, 1, 1)
        plt.plot(pulse.X, intensity, color = "darkorange")
        plt.plot(pulse.X, pulse.Y, color = "black", linestyle = "dashed")
        plt.grid()
        plt.legend(["Evolved intensity", "Initial intensity"])
        plt.title("Spectral intensity")
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
    array = torch.tensor([[np.real(array[i]), 0] for i in range(len(array))], requires_grad = True, device = device, dtype = dtype)
    array = torch.view_as_complex(array)
    array = array.reshape(1, array.numel())

    return array


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def comp_var(X, Y):
    '''
    Variance of the distribution.
    '''
    Y_2 = Y.copy()
    Y_2 = np.abs(Y_2)
    Y_2 /= np.sum(Y_2)
    X_mean = np.mean(X)
    var = np.sum(Y_2*(X-X_mean)**2)
    return var


def comp_std(X, Y):
    '''
    Standard deviation of the distribution.
    '''
    return np.sqrt(comp_var(X, Y))


def comp_FWHM(std):
    '''
    Estimate Full Width at Half Maximum given the standard deviation of the distribution. For a gaussian the formula is precise.
    '''
    return 2*np.sqrt(2*np.log(2))*std


def comp_mean_TBP(initial_X, initial_FWHM):
    FWHMs = []
    intensity_labels = os.listdir('data/train_intensity')
    for label in tqdm(intensity_labels):
        intensity = np.loadtxt("data/train_intensity/" + label)
        FWHMs.append(comp_FWHM(comp_std(initial_X, intensity)))

    TBPs = np.array(FWHMs)*initial_FWHM/2
    return np.mean(TBPs), np.std(TBPs)


def shift_to_centre(intensity_to_shift, intensity_ref):
    '''
    Return the \"intensity_to_shift\" shifted in such a way that its center of mass is on the same index as in the case of the \"intensity_ref\".
    '''
    if len(intensity_to_shift) != len(intensity_ref):
        raise Exception("Both intensities must be of equal length.")
    num = len(intensity_ref)

    x_axis = np.linspace(1, 2, num) # doesn't matter what we take here; I just want to create spectrum
    spectrum_to_shift = sa.spectrum(x_axis, intensity_to_shift, "freq", "intensity")
    spectrum_ref = sa.spectrum(x_axis, intensity_ref, "freq", "intensity")

    com_s = spectrum_to_shift.comp_center()
    com_r = spectrum_ref.comp_center()

    spectrum_to_shift.very_smart_shift(com_s-com_r, inplace = True)
    return np.real(spectrum_to_shift.Y)