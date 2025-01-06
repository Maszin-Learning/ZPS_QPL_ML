import numpy as np
import matplotlib.pyplot as plt
from math import floor
import os
import spectral_analysis as sa
from utilities import np_to_complex_pt, evolve_np, evolve_pt, shift_to_centre, wl_to_freq, freq_to_wl, complex_intensity
import utilities as u
from torch.nn import MSELoss
import torch
from scipy.interpolate import CubicSpline
from scipy.interpolate import splrep, BSpline
from torch.fft import ifft, ifftshift

def test(model, 
         target_pulse, 
         initial_pulse, 
         device, 
         dtype, 
         save, 
         param,
         iter_num = 0):
    '''
    # OLD LEGEND

    ## Test how the model transforms initial pulse to the target pulse.
    # Arguments:

    model - the model of the neural network.

    target_pulse - one-dimensional complex Pytorch Tensor.

    initial_pulse - a spectrum class object

    iter_num - the test plot is saved as \"pics/reconstructed_[iter_num].jpg\"

    # Returns:

    (plot, loss) - where plot (returned in a strange way) depicts model predictions on target pulse and phase, 
    and loss is MSE of that prediction.

    # Note: initial_pulse_Y, initial_pulse_X and target_pulse must have the same length.
    '''

    # prepare targets
    temp_intens_target = target_pulse.clone()
    temp_intens_target = torch.tensor(temp_intens_target, requires_grad = False, device = device, dtype = dtype)  # well, it was a tensor even before, but now we know its properties

    spectr_intens_target = u.fourier(temp_intens_target)
    spectr_intens_target = u.cut(spectr_intens_target, param.freq_width/param.init_freq_res) # we leave central 100 GHz, we delete the rest in order to save GPU
    spectr_intens_target = u.increase_resolution(spectr_intens_target, param.increase_freq_res, device = device, dtype = dtype)

    # generate phases
    temp_phase_pred, spectr_phase_pred = model(target_pulse)

    # we apply temporal phase
    temp_phase_pred = u.increase_resolution(temp_phase_pred, param.eopm_res/param.comp_time_res, device = device, dtype = dtype) # 11 ps is the resolution of EOPM
    initial_intensity_pt = u.np_to_complex_pt(initial_pulse.Y, device = device, dtype = dtype)
    temp_intens_pred = u.multiply_by_phase(initial_intensity_pt, temp_phase_pred, index_start = param.temp_idx_start, device = device, dtype = dtype)

    # we apply spectral phase
    spectr_intens_pred = u.fourier(temp_intens_pred)

    old_length = np.array(spectr_intens_pred.shape)[-1]
    spectr_intens_pred = u.cut(spectr_intens_pred, param.freq_width/param.init_freq_res) # we leave central 100 GHz, we delete the rest in order to save GPU
    new_length = np.array(spectr_intens_pred.shape)[-1]
    increase_time_res = old_length/new_length
    print(torch.sum(torch.mul(spectr_intens_pred, torch.conj(spectr_intens_pred))))   
    spectr_intens_pred = u.increase_resolution(spectr_intens_pred, param.increase_freq_res, device = device, dtype = dtype)
    spectr_phase_pred = u.increase_resolution(spectr_phase_pred, param.pulse_shaper_res/param.comp_freq_res, device = device, dtype = dtype)  # 1.5 GHz is the resolution of the pulse shaper
    print(torch.sum(torch.mul(spectr_intens_pred, torch.conj(spectr_intens_pred))))    
    spectr_intens_pred = u.multiply_by_phase(spectr_intens_pred, spectr_phase_pred, index_start = floor((spectr_intens_pred.shape[-1]-spectr_phase_pred.shape[-1])/2), device = device, dtype = dtype)

    spectr_X = np.array([-param.freq_width*1000/2 + param.comp_freq_res*1000*n for n in range(spectr_intens_pred.shape[-1])])# this is in GHz!!!

    # and back to time domain
    print(torch.sum(torch.mul(spectr_intens_pred, torch.conj(spectr_intens_pred))))
    temp_intens_pred2 = u.inv_fourier(spectr_intens_pred)
    temp_intens_pred2 = u.increase_resolution(temp_intens_pred2, increase_time_res, device = device, dtype = dtype)
    print(torch.sum(torch.mul(temp_intens_pred2, torch.conj(temp_intens_pred2))))
    temp_intens_pred2 = u.cut(temp_intens_pred2, np.array(len(target_pulse)))

    # create plots

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    axes[1, 1].axis('off')

    # plot 1
    ax1.plot(initial_pulse.X, np.abs(initial_pulse.Y)**2, color="red", zorder = 10, lw = 2)     
    ax1.plot(initial_pulse.X, (temp_intens_target.clone().detach().cpu().numpy().flatten())**2, color = "blue", alpha = 0.5, lw =5, zorder = 0)            
    ax1.set_title("Step 1")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Normalized intensity")
    ax1.grid()

    ax1_ph = plt.twinx(ax1) # ax1 for the phase
    temp_idx_end = np.array(temp_phase_pred.shape)[-1] + param.temp_idx_start   # we want to find the indices of the interval in
    ax1_ph.plot(initial_pulse.X[param.temp_idx_start: temp_idx_end],
                 np.real(temp_phase_pred.clone().detach().cpu().numpy()),
                   linestyle = "dashed", color = "darkorange", zorder = 5)
    ax1.legend(["You are a dick", "And you a cunt"])

    # plot 2
    ax2.plot(spectr_X, np.abs(spectr_intens_pred.clone().detach().cpu().numpy().flatten())**2, color="red", zorder = 10, lw =2)
    ax2.plot(spectr_X, np.abs(spectr_intens_target.clone().detach().cpu().numpy().flatten())**2, color = "darkorange", alpha = 0.7, lw = 5, zorder = 0)
    ax2.set_title("Step 2")
    ax2.set_xlabel("Frequency (THz)")
    ax2.set_ylabel("Normalized intensity")
    ax2.grid()

    # plot 3
    ax3.plot(initial_pulse.X, np.abs(temp_intens_pred2.clone().detach().cpu().numpy().flatten())**2, color="red", lw = 2)
    ax3.plot(initial_pulse.X, (temp_intens_target.clone().detach().cpu().numpy().flatten())**2, color = "blue", alpha = 0.5, lw =5, zorder = 0)            
    ax3.set_title("Step 3")
    ax3.set_xlabel("Time (ps)")
    ax3.set_ylabel("Normalized intensity")
    ax3.grid()

    # save the figure if needed
    if save:
        if not os.path.isdir("pics"):
            os.mkdir("pics")
        fig.savefig(f"pics/reconstructed_{iter_num}.svg", bbox_inches="tight", dpi=200)

    return fig, 0


def create_initial_pulse(bandwidth, centre, FWHM, num, pulse_type):

    if pulse_type == "gauss":
        pulse = sa.hermitian_pulse(pol_num = 0,
                                    bandwidth = bandwidth,
                                    centre = centre,
                                    FWHM = FWHM,
                                    num = num,
                                    x_type = "time")
        pulse.Y = np.abs(pulse.Y)
        return pulse
    
    elif pulse_type == "hermite":
        pulse = sa.hermitian_pulse(pol_num = 1,
                                    bandwidth = bandwidth,
                                    centre = centre,
                                    FWHM = FWHM,
                                    num = num,
                                    x_type = "time")
        pulse.Y = np.abs(pulse.Y)
        return pulse
    
    elif pulse_type == "exponential":

        Y = np.flip(np.exp(1*np.linspace(-10, 3, num)) - np.exp(-10))
        for i in range(0, floor(1/3*num)):
            Y[i] = 0
        Y = Y/np.max(np.abs(Y))
        Y = np.roll(Y, floor(600*num/5000)) # 600 shifts to center for num=5000
        X = np.linspace(bandwidth[0], bandwidth[1], num)
        pulse = sa.spectrum(X = X, Y = Y, x_type ="time", y_type ="intensity")
        pulse.Y = np.abs(pulse.Y)
        return pulse
    
    else:
        raise Exception("Pulse_type must be either \"gauss\", \"hermite\" or \"exponential\".")