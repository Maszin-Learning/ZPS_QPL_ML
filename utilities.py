import torch
from torch import nn
import numpy as np
from math import floor, ceil
import matplotlib.pyplot as plt
import spectral_analysis as sa
import os
import shutil
from tqdm import tqdm
from torch.fft import fft, fftshift
from numba import jit
    
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
    
    compl_intensity = torch.mul(intensity, torch.exp(1j*long_phase))

    compl_intensity = torch.fft.ifftshift(compl_intensity)
    compl_intensity = torch.fft.ifft(compl_intensity)
    compl_intensity = torch.fft.ifftshift(compl_intensity)

    if abs:
        return compl_intensity.abs()
    else:
        return compl_intensity
    
def complex_intensity(intensity, phase, device, dtype):
    '''
    Given Pytorch \"intensity\" and \"phase\", we want to multiply them with each other (exp of phase).
    Only problem is that the \"phase\" is much shorter than \"intensity\".
    '''

    zeroes_shape = np.array(phase.shape)
    zeroes_shape[-1] = floor((intensity.shape[-1]-phase.shape[-1])/2)
    zeroes_shape = tuple(zeroes_shape)

    long_phase = torch.concat([torch.zeros(size = zeroes_shape, requires_grad = True, device = device, dtype = dtype), 
                          phase,
                          torch.zeros(size = zeroes_shape, requires_grad = True, device = device, dtype = dtype)], dim=phase.ndim-1)
    
    return torch.mul(intensity, torch.exp(1j*long_phase))
    
    
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

        intensity = np.loadtxt("data/train_intensity/" + intensity_labels[i])

        plt.subplot(2, 1, 2)
        plt.fill_between(pulse_safe.X + 375, np.real(pulse_safe.Y), color = "orange", alpha = 0.4)
        plt.grid()
        plt.legend(["Spectral intensity"])
        plt.title("Train phase")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Spectral phase (rad)")

        plt.subplot(2, 1, 1)
        plt.plot(pulse.X, intensity, color = "darkorange")
        plt.plot(pulse.X, pulse.Y, color = "black", linestyle = "dashed")
        plt.grid()
        plt.legend(["Evolved intensity", "Initial intensity"])
        plt.title("Temporal intensity")
        plt.xlabel("Time (ps)")
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
    spectrum_to_shift = sa.spectrum(x_axis, intensity_to_shift, "THz", "intensity")
    spectrum_ref = sa.spectrum(x_axis, intensity_ref, "THz", "intensity")

    com_s = spectrum_to_shift.comp_center(norm = "L2")
    com_r = spectrum_ref.comp_center(norm = "L2")
    spectrum_to_shift.smart_shift(com_r-com_s, inplace = True)
    return np.abs(spectrum_to_shift.Y)


def integrate(intensity):
    return np.sum(intensity*np.conjugate(intensity))

    
def unwrap(x):
    ''''''
    x_1 = 0
    
    if len(x.shape) == 1:
        for i in range(x.shape[-1]):
            _x_1 = x[i]
            if _x_1 - x_1> np.pi:
                x[i] -=2*np.pi 
            if _x_1 - x_1< -np.pi:
                x[i] +=2*np.pi
            x_1 = _x_1

    elif len(x.shape) == 2:
        for batch in range(x.shape[0]):              
            for i in range(x.shape[-1]):
                _x_1 = x[batch,i]
                if _x_1 - x_1> np.pi:
                    x[batch,i] -=2*np.pi 
                if _x_1 - x_1< -np.pi:
                    x[batch,i] +=2*np.pi
                x_1 = _x_1

    else:
        raise Exception("The \"x\" has surprisingly many dimensions...")
    return x

def clear_folder(name:str):
    #saving model
    if os.path.isdir(name):
        shutil.rmtree(name)
        os.mkdir(name)
    else:
        os.mkdir(name)

def wl_to_freq(wl):
    if type(wl) == type(np.array([])):
        return 299792458/np.flip(wl)/1000
    else:
        return 299792458/wl/1000

def freq_to_wl(freq):
    if type(freq) == type(np.array([])):
        return 299792458/np.flip(freq)/1000
    else:
        return 299792458/freq/1000
    
def increase_resolution(pt_tensor, times, device, dtype, keep_norm = True):
    '''
    Returns initial tensor with resolution increased \"times\" times.
    '''
    if times < 1:
        raise Exception("\"times\" must be greater or equal to 1.")
    long_tensor = pt_tensor.clone()

    long_tensor = torch.fft.fftshift(long_tensor)
    long_tensor = torch.fft.fft(long_tensor, norm = "ortho")
    long_tensor = torch.fft.fftshift(long_tensor)

    zeroes_shape = np.array(long_tensor.shape)
    zeroes_shape[-1] = floor((times*zeroes_shape[-1])/2) # if we want to increase resolution 10 times, we need to pad 10 times more zero than the length of the signal
    zeroes_shape = tuple(zeroes_shape)
    long_tensor = torch.concat([torch.zeros(size = zeroes_shape, requires_grad = True, device = device, dtype = dtype), 
                          long_tensor,
                          torch.zeros(size = zeroes_shape, requires_grad = True, device = device, dtype = dtype)], dim=long_tensor.ndim-1)

    long_tensor = torch.fft.ifftshift(long_tensor)
    long_tensor = torch.fft.ifft(long_tensor, norm = "ortho")
    long_tensor = torch.fft.ifftshift(long_tensor)

    return long_tensor

def multiply_by_phase(intensity, phase, index_start, device, dtype):

    zeroes_shape_left = np.array(phase.shape)
    zeroes_shape_right = np.array(phase.shape)
    zeroes_shape_left[-1] = index_start
    zeroes_shape_right[-1] = np.array(intensity.shape)[-1] - index_start - np.array(phase.shape)[-1]

    if zeroes_shape_right[-1] < 0:
        raise Exception("Cannot multiply intensity by the phase ({}), because it is longer than intensity ({}).".format(phase.shape[-1], intensity.shape[-1]))
    
    zeroes_shape_left = tuple(zeroes_shape_left)
    zeroes_shape_right = tuple(zeroes_shape_right)

    long_phase = torch.concat([torch.zeros(size = zeroes_shape_left, requires_grad = True, device = device, dtype = dtype), 
                          phase,
                          torch.zeros(size = zeroes_shape_right, requires_grad = True, device = device, dtype = dtype)], 
                          dim=phase.ndim-1)

    return torch.mul(intensity, torch.exp(1j*long_phase))

def cut(pt_array, num):
    '''
    Delete both \"tails\" of the array, leaving only \"num\" points at the center
    '''

    size = pt_array.shape[-1]
    if len(pt_array.shape) == 1:
        return pt_array[floor(size/2-num/2):ceil(size/2+num/2)]
    elif len(pt_array.shape) == 2:
        return pt_array[:,floor(size/2-num/2):ceil(size/2+num/2)]
    elif len(pt_array.shape) == 3:
        return pt_array[:,:,floor(size/2-num/2):ceil(size/2+num/2)]
    else:
        raise Exception("WTF, the shape of this tensor is wild.")
    
def soft_abs(x, epsilon=1e-10):
    return torch.sqrt(x*torch.conj(x) + epsilon)
    
def fourier(tensor, keep_norm = False):
    tensor2 = tensor.clone()
    if keep_norm:
        tensor2 = torch.mul(torch.sqrt(soft_abs(tensor2)), torch.exp(1j*tensor2.angle()))
    tensor2 = torch.fft.fftshift(tensor2)
    tensor2 = torch.fft.fft(tensor2, norm = "ortho")
    tensor2 = torch.fft.fftshift(tensor2)
    if keep_norm:
        tensor2 = tensor2*soft_abs(tensor2)
    return tensor2

def inv_fourier(tensor, keep_norm = False):
    tensor2 = tensor.clone()
    if keep_norm:
        tensor2 = torch.mul(torch.sqrt(soft_abs(tensor2)), torch.exp(1j*tensor2.angle()))
    tensor2 = torch.fft.ifftshift(tensor2)
    tensor2 = torch.fft.ifft(tensor2, norm = "ortho")
    tensor2 = torch.fft.ifftshift(tensor2)
    if keep_norm:
        tensor2 = tensor2*soft_abs(tensor2)
    return tensor2


class Parameters:
    '''
    ## Storage for some metaparameters.

    comp_time_resolution - time domain resolution used for computation. It's much higher than the real one to avoid numerical errors.

    comp_freq_resolution - frequency domain resolution used for computation. Higher than the \"physical\" one.

    init_freq_resolution - frequency domain resolution determined by the signal properties in time. Cannot be arbitrary defined.

    temp_idx_start - index of the \"first\" non-zero intensity point in the time domain with respect to comp_time_resolution. Starting
    with this index we multiply intensity by the phase.

    increase_freq_res - comp_freq_resolution/init_freq_resolution
    '''

    def __init__(self):
        self.comp_time_res = None 
        self.temp_idx_start = None
        self.init_freq_res = None
        self.increase_freq_res = None
        self.comp_freq_res = None
        self.eopm_res = None
        self.pulse_shaper_res = None
        self.freq_width = None

# debugging functions

def plot(tensor):
    X = np.array(range(np.array(tensor.shape)[-1]))
    Y = np.abs(tensor.clone().detach().cpu().numpy())
    plt.scatter(X, Y, color = "red", s = 1)
    plt.grid()
    plt.show()

def plot2(tensor1, tensor2):
    X = np.array(range(np.array(tensor1.shape)[-1]))
    Y1 = np.abs(tensor1.clone().detach().cpu().numpy())
    Y2 = np.abs(tensor2.clone().detach().cpu().numpy())
    plt.scatter(X, Y1, color = "red", s = 1)
    plt.scatter(X, Y2, color = "blue", s = 1)
    plt.grid()
    plt.show()

def print_gradient_after(tensor, name = "So far"):
    """
    Registers a backward hook to log gradient norms after backpropagation.
    """
    def hook_fn(grad):
        grad_norm = grad.abs().max().item()
        print(f"{name} | Max Gradient Norm: {grad_norm}")
    print(" Computing gradient norm...")
    tensor.register_hook(hook_fn)

def power(tensor):
    return torch.sum(torch.square(torch.abs(tensor)), axis = -1).detach().cpu().numpy()