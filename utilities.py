import torch
from torch import nn
import numpy as np
from math import floor
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
    spectrum_to_shift = sa.spectrum(x_axis, intensity_to_shift, "freq", "intensity")
    spectrum_ref = sa.spectrum(x_axis, intensity_ref, "freq", "intensity")

    com_s = spectrum_to_shift.comp_center(norm = "L2")
    com_r = spectrum_ref.comp_center(norm = "L2")
    spectrum_to_shift.smart_shift(com_r-com_s, inplace = True)
    return np.abs(spectrum_to_shift.Y)


def integrate(intensity):
    return np.sum(intensity*np.conjugate(intensity))
 

### FILTER FUNCTIONS ###


def gen_filter_mask(threshold, device, steepness = 10, num = 5000):
    '''
    ***steepness*** - an integer positive number denoting the steepness of th filter threshold; the bigger the steeper
    ***threshold*** - a fraction of the whole frequency spectrum that goes THROUGH the filter (for the low pass filter) or gets BLOCKED by the filter (high pass)
    ***num*** - length of the filter mask 

    The mask is basically a vector of length ***num*** with ***threshold*** * ***num*** ones at the center at zeroes at the sides.
    '''
    X = np.linspace(-2, 2, num)
    if threshold == 0:
        output = np.zeros(num)
    elif threshold == 1:
        output = np.ones(num)
    else:
        output = np.exp(-(X/(2*threshold))**(2*steepness))
    output = torch.from_numpy(output)
    output = output.to(device)
    return output

def low_pass_pt(signal, filter_mask):

    signal2 = signal.clone()
    signal2 = torch.fft.fftshift(signal2)
    signal2 = torch.fft.fft(signal2)
    signal2 = torch.fft.fftshift(signal2)

    signal2 = signal2*filter_mask

    signal2 = torch.fft.ifftshift(signal2)
    signal2 = torch.fft.ifft(signal2)
    signal2 = torch.fft.ifftshift(signal2)

    signal2 = signal2.real
    signal2 = signal2.to(signal.dtype)
    return signal2

def high_pass_pt(signal, filter_mask):

    signal2 = signal.clone()
    signal2 = torch.fft.fftshift(signal2)
    signal2 = torch.fft.fft(signal2)
    signal2 = torch.fft.fftshift(signal2)

    signal2 = signal2*(1-filter_mask)

    signal2 = torch.fft.ifftshift(signal2)
    signal2 = torch.fft.ifft(signal2)
    signal2 = torch.fft.ifftshift(signal2)

    signal2 = signal2.real
    signal2 = signal2.to(signal.dtype)
    return signal2

def low_pass_np(signal, filter_mask):
    signal2 = signal.copy()
    signal2 = np.fft.fftshift(signal2)
    signal2 = np.fft.fft(signal2)
    signal2 = np.fft.fftshift(signal2)

    signal2 = signal2*filter_mask.clone().detach().cpu().numpy()

    signal2 = np.fft.ifftshift(signal2)
    signal2 = np.fft.ifft(signal2)
    signal2 = np.fft.ifftshift(signal2)
    return np.real(signal2)

def high_pass_np(signal, filter_mask):
    signal2 = signal.copy()
    signal2 = np.fft.fftshift(signal2)
    signal2 = np.fft.fft(signal2)
    signal2 = np.fft.fftshift(signal2)

    signal2 = signal2*(1-filter_mask.clone().detach().cpu().numpy())

    signal2 = np.fft.ifftshift(signal2)
    signal2 = np.fft.ifft(signal2)
    signal2 = np.fft.ifftshift(signal2)
    return np.real(signal2)

def diff_pt(vector, device, dtype):
    zero_shape = np.array(torch.diff(vector).shape)
    zero_shape[-1] = 1
    zero_shape = tuple(zero_shape)
    return torch.cat([torch.zeros(zero_shape, device = device, dtype = dtype),torch.diff(vector)], dim=torch.diff(vector).ndim-1)


### LOSS FUNCTIONS


class MSEsmooth(nn.modules.loss._Loss):
    '''
    Classical MSE loss function with penalty for rapid changes of the transforming phase.
    \"c_factor\" denotes ratio of the penalty to the MSE.
    '''
    
    def __init__(self, device, dtype, c_factor = 0.6):
        super(MSEsmooth, self).__init__()
        self.c_factor = c_factor
        self.device = device
        self.dtype = dtype

    def forward(self, results, target):

        pred_phase, pred_intensity = results

        MSE_sum = torch.sum(torch.square(pred_intensity - target))

        zero_shape = np.array(torch.diff(pred_phase).shape)
        zero_shape[-1] = 1
        zero_shape = tuple(zero_shape)

        cont_penalty = torch.mean(torch.square(diff_pt(unwrap(pred_phase), device = self.device, dtype = self.dtype)))
        cont_penalty = cont_penalty/(cont_penalty.clone().detach())
        cont_penalty = self.c_factor*cont_penalty*(MSE_sum.clone().detach())

        return MSE_sum + cont_penalty
    
class MSEsmooth2(nn.modules.loss._Loss):
    '''
    Classical MSE loss function with continuity and smoothness penalty for rapid changes of the transforming phase.
    \"c_factor\" denotes ratio of the continuity penalty to the MSE, while \"s_factor"\" denotes the analogous ratio in case
    of smoothness penalty.
    '''
    
    def __init__(self, device, dtype, c_factor = 0.6, s_factor = 0.2):
        super(MSEsmooth2, self).__init__()
        self.c_factor = c_factor
        self.s_factor = s_factor
        self.device = device
        self.dtype = dtype

    def forward(self, results, target):

        pred_phase, pred_intensity = results

        MSE_sum = torch.sum(torch.square(pred_intensity - target))

        zero_shape = np.array(torch.diff(pred_phase).shape)
        zero_shape[-1] = 1
        zero_shape = tuple(zero_shape)

        phase_unwraped = unwrap(pred_phase)

        cont_penalty = torch.mean(torch.square(diff_pt(phase_unwraped, device = self.device, dtype = self.dtype)))
        cont_penalty = cont_penalty/(cont_penalty.clone().detach())
        cont_penalty = self.c_factor*cont_penalty*(MSE_sum.clone().detach())

        smooth_penalty = torch.mean(torch.square(diff_pt(diff_pt(phase_unwraped, device = self.device, dtype = self.dtype), device = self.device, dtype = self.dtype)))
        smooth_penalty = smooth_penalty/(smooth_penalty.clone().detach())
        smooth_penalty = self.s_factor*smooth_penalty*(MSE_sum.clone().detach())

        return MSE_sum + cont_penalty + smooth_penalty
    
class MSElowpass(nn.modules.loss._Loss):
    '''
    Classical MSE loss function with penalty for high frequency components of the phase.
    '''
    
    def __init__(self, device, dtype, filter_mask, penalty_strength):
        super(MSElowpass, self).__init__()
        self.filter_mask = filter_mask
        self.penalty_strength = penalty_strength
        self.device = device
        self.dtype = dtype

    def forward(self, results, target):

        pred_phase, pred_intensity = results

        MSE_sum = torch.sum(torch.square(pred_intensity - target))

        pred_phase2 = pred_phase.clone()
        pred_phase2 = torch.fft.fftshift(pred_phase2)
        pred_phase2 = torch.fft.fft(pred_phase2)
        pred_phase2 = torch.fft.fftshift(pred_phase2)
        fast_penalty = (torch.ones(self.filter_mask.shape, device = self.device) - self.filter_mask)*pred_phase2
        fast_penalty = fast_penalty*torch.conj_physical(fast_penalty)
        fast_penalty = fast_penalty.real
        fast_penalty = torch.mean(fast_penalty)
        fast_penalty = fast_penalty/(fast_penalty.clone().detach())
        fast_penalty = self.penalty_strength*fast_penalty*(MSE_sum.clone().detach())

        return MSE_sum + fast_penalty
    
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