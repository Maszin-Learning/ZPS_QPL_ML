import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import floor, ceil
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
    #signal_2_temp = target_pulse.clone()
    #signal_2_temp = torch.tensor(signal_2_temp, requires_grad = False, device = device, dtype = dtype)  # well, it was a tensor even before, but now we know its properties

    # generate phases
    temp_phase_pred, spectr_phase_pred = model(target_pulse)

    # we apply temporal phase
    temp_phase_pred = u.increase_resolution(temp_phase_pred, param.eopm_res/param.comp_time_res, device = device, dtype = dtype) # 11 ps is the resolution of EOPM
    initial_intensity_pt = u.np_to_complex_pt(initial_pulse.Y, device = device, dtype = dtype)
    signal_1_temp = u.multiply_by_phase(initial_intensity_pt, temp_phase_pred, index_start = param.temp_idx_start, device = device, dtype = dtype)

    signal_1_spectr = u.fourier(signal_1_temp)
    signal_1_spectr = u.cut(signal_1_spectr, param.freq_width/param.init_freq_res) # we leave central 100 GHz, we delete the rest in order to save GPU
    signal_1_spectr = u.increase_resolution(signal_1_spectr, param.increase_freq_res, device = device, dtype = dtype)

    # we apply spectral phase
    signal_2_spectr = u.fourier(target_pulse.clone())

    old_length = np.array(signal_2_spectr.shape)[-1]
    signal_2_spectr = u.cut(signal_2_spectr, param.freq_width/param.init_freq_res) # we leave central 100 GHz, we delete the rest in order to save GPU

    new_length = np.array(signal_2_spectr.shape)[-1]
    increase_time_res = old_length/new_length

    signal_2_spectr = u.increase_resolution(signal_2_spectr, param.increase_freq_res, device = device, dtype = dtype)
    spectr_phase_pred = u.increase_resolution(spectr_phase_pred, param.pulse_shaper_res/param.comp_freq_res, device = device, dtype = dtype)  # 1.5 GHz is the resolution of the pulse shaper
    spectr_idx_start = floor((signal_2_spectr.shape[-1]-spectr_phase_pred.shape[-1])/2)
    signal_2_spectr = u.multiply_by_phase(signal_2_spectr, spectr_phase_pred, index_start = spectr_idx_start, device = device, dtype = dtype)
    spectr_X = np.array([-param.freq_width*1000/2 + param.comp_freq_res*1000*n for n in range(signal_2_spectr.shape[-1])])# this is in GHz!!!

    # and back to time domain
    signal_2_temp = u.inv_fourier(signal_2_spectr)
    signal_2_temp = u.increase_resolution(signal_2_temp, increase_time_res, device = device, dtype = dtype)
    signal_2_temp = u.cut(signal_2_temp, np.array(len(target_pulse)))

    # create plots

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    # plot 1
    ax1.plot(initial_pulse.X, np.abs(initial_pulse.Y)**2, color="red", zorder = 10, lw = 2)     
    ax1.plot(initial_pulse.X, (target_pulse.clone().detach().cpu().numpy().flatten())**2, color = "blue", alpha = 0.5, lw =5, zorder = 0)            
    ax1.set_title("Step 1: Applying temporal phase to the first signal")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Normalized intensity")
    ax1.set_xlim([-1000, 2000])
    ax1.grid()
    
    # legend and phase for ax1

    ax1_ph = plt.twinx(ax1) # ax1 for the phase
    temp_idx_end = np.array(temp_phase_pred.shape)[-1] + param.temp_idx_start   # we want to find the indices of the interval in

    x = [initial_pulse.X[param.temp_idx_start: temp_idx_end][0]]
    y = [np.real(temp_phase_pred.clone().detach().cpu().numpy())[0]]
    ax1_ph.plot(x, y, color="red", zorder = 10, lw = 2)     
    ax1_ph.plot(x, y, color = "blue", alpha = 0.5, lw =5, zorder = 0)    
    ax1_ph.plot(initial_pulse.X[param.temp_idx_start: temp_idx_end],
                 np.unwrap(np.real(temp_phase_pred.clone().detach().cpu().numpy())),
                   linestyle = "dashed", color = "darkorange", zorder = 1)
    
    ax1_ph.legend(["Unmodified 1st signal", "Unmodified 2nd signal", "Temporal phase in EOPM"], 
                        facecolor="white", framealpha=1, loc="upper right")

    # plot 2
    xlim = [-15, 15]
    ax2.plot(spectr_X, np.abs(signal_1_spectr.clone().detach().cpu().numpy().flatten())**2, color="red", zorder = 10, lw =2)
    ax2.plot(spectr_X, np.abs(signal_2_spectr.clone().detach().cpu().numpy().flatten())**2, color = "darkorange", alpha = 0.7, lw = 5, zorder = 0)
    ax2.set_title("Step 2: Applying spectral phase to the second signal")
    ax2.set_xlabel("Frequency around centre (GHz)")
    ax2.set_ylabel("Normalized intensity")
    ax2.set_xlim(xlim)
    ax2.grid()
    
    spectr_X_ph = np.linspace(np.mean(spectr_X) - 1000*(param.pulse_shaper_res*param.spectral_phase_len/2), 
                              np.mean(spectr_X) + 1000*(param.pulse_shaper_res*param.spectral_phase_len/2),
                              spectr_phase_pred.shape[-1])
    spectr_X_ph = spectr_X_ph[np.searchsorted(spectr_X_ph, xlim[0]): np.searchsorted(spectr_X_ph, xlim[1])]

    ax2_ph = plt.twinx(ax2) # ax2 for the phase

    # ax2 phase plot + legend
    x = [spectr_X_ph[0]]
    y = [spectr_phase_pred.clone().detach().cpu().numpy()[np.searchsorted(spectr_X_ph, xlim[0]): np.searchsorted(spectr_X_ph, xlim[1])][0]]
    ax2_ph.plot(x, y, color="red", zorder = 10, lw = 2)     
    ax2_ph.plot(x, y, color = "darkorange", alpha = 0.7, lw = 5, zorder = 0)       

    ax2_ph.plot(spectr_X_ph,
                 spectr_phase_pred.clone().detach().cpu().numpy()[np.searchsorted(spectr_X_ph, xlim[0]): np.searchsorted(spectr_X_ph, xlim[1])], 
                 linestyle = "dashed", color = "green", zorder = 0)
    ax2_ph.legend(["Modified 1st signal", "Unmodified 2nd signal", "Spectral phase in P-Sh"],
                                          facecolor="white", framealpha=1, loc="upper right")

    # plot 3

    ax3.plot(initial_pulse.X, np.abs(signal_1_temp.clone().detach().cpu().numpy().flatten())**2, color = "red", lw = 2, zorder = 0)            
    ax3.plot(initial_pulse.X, np.abs(signal_2_temp.clone().detach().cpu().numpy().flatten())**2, color = "blue", alpha = 0.5, lw = 5)    
    ax3.set_title("Step 3: Both signals in the time domain")
    ax3.set_xlabel("Time (ps)")
    ax3.set_ylabel("Normalized intensity")
    ax3.grid()
    ax3.set_xlim([-1000, 1500])

    # phase of ax3 and legend

    ax3_ph = plt.twinx(ax3)
    idx_sp_ph_start = np.searchsorted(initial_pulse.X, -150)
    idx_sp_ph_end = np.searchsorted(initial_pulse.X, 550)

    x = [initial_pulse.X[idx_sp_ph_start:idx_sp_ph_end][0]]
    y = [np.angle(signal_2_temp.clone().detach().cpu().numpy().flatten())[idx_sp_ph_start:idx_sp_ph_end][0]]

    ax3_ph.plot(x, y, color = "red", lw = 2) 
    ax3_ph.plot(x, y, color = "blue", lw = 5, alpha = 0.5, zorder = 0)            
    ax3_ph.plot(initial_pulse.X[idx_sp_ph_start:idx_sp_ph_end],
                 np.angle(signal_1_temp.clone().detach().cpu().numpy().flatten())[idx_sp_ph_start:idx_sp_ph_end], 
                 color = "green", alpha = 1, linestyle = "dashed")      
    ax3_ph.plot(initial_pulse.X[idx_sp_ph_start:idx_sp_ph_end],
                 np.angle(signal_2_temp.clone().detach().cpu().numpy().flatten())[idx_sp_ph_start:idx_sp_ph_end], 
                 color = "darkorange", alpha = 1, linestyle = "dashed")   
    ax3_ph.legend(["1st signal", "2nd signal", "1st signal's phase", "2nd signal's phase"],
                                        facecolor="white", framealpha=1, loc="upper right")      
    
    # statistics

    t_target = signal_2_temp.clone().detach().cpu().numpy().flatten()
    t_pred = signal_1_temp.clone().detach().cpu().numpy().flatten()
    s_target = signal_2_spectr.clone().detach().cpu().numpy().flatten()
    s_pred = signal_1_spectr.clone().detach().cpu().numpy().flatten()
    initial = initial_pulse.Y

    init_power = "\nInitial power: " + str(round(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)), 5))
    trg_power = "\nTarget power: " + str(round(np.sum(t_target*np.conjugate(t_target)), 5))
    pred_power = "\nPrediction power: " + str(round(np.real(np.sum(t_pred*np.conjugate(t_pred))), 5))
    
    t_MSE = np.sum(np.abs(t_target-t_pred)**2)
    s_MSE = np.sum(np.abs(s_target-s_pred)**2)
    all_MSE = t_MSE + s_MSE

    temp_MSE = "\n\nTemporal MSE: " + str(t_MSE)
    spectr_MSE = "\nSpectral MSE: " + str(s_MSE)
    tot_MSE = "\nTotal MSE: " + str(all_MSE)

    init_hom_value =  np.abs(np.sum(initial*np.conjugate(t_target))*np.sum(np.conjugate(initial)*t_target))
    final_hom_value = np.abs(np.sum(t_target*np.conjugate(t_pred))*np.sum(np.conjugate(t_target)*t_pred)) # abs to kill 0j
    #final_hom_value = np.abs(np.sum(t_target*np.conjugate(t_target))*np.sum(np.conjugate(t_target)*t_target)) # abs to kill 0j

    init_hom = "\n\nInitial HOM visibility: " + str(round(100*init_hom_value, 1)) + "%"
    final_hom = "\nFinal HOM visibility: " + str(round(100*final_hom_value, 1)) + "%"

    ax4.axis('off')
    ax4.text(x = 0, y = 0.5, 
             s = "STATISTICS:\n" + init_power + trg_power + pred_power + temp_MSE + spectr_MSE + tot_MSE + init_hom + final_hom,
             transform = ax4.transAxes)
    
    # save the figure if needed
    if save:
        if not os.path.isdir("pics"):
            os.mkdir("pics")
        fig.savefig(f"pics/reconstructed_{iter_num}.svg", bbox_inches="tight", dpi=1600)

    return fig, 0, round(100*final_hom_value, 1)


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