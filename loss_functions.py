import numpy as np
import torch
from utilities import unwrap


### FILTER FUNCTIONS - necessary to define the penalty for high frequencies ###


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
    
class MSEdouble(nn.modules.loss._Loss):
    '''
    Sum of MSE applied to both intensities.
    '''
    
    def __init__(self, device, dtype):
        super(MSEdouble, self).__init__()

        self.device = device
        self.dtype = dtype

    def forward(self, temp_phase_pred, spectr_phase_pred, temp_intens_pred, spectr_intens_pred, temp_intens_target, spectr_intens_target):

        MSE_t = torch.sum(torch.square(temp_intens_pred - temp_intens_target))
        MSE_s = torch.sum(torch.square(spectr_intens_pred - spectr_intens_target))

        return MSE_t+MSE_s