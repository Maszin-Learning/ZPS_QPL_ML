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
    temp_intens_target = target_pulse.clone()
    temp_intens_target = torch.tensor(temp_intens_target, requires_grad = False, device = device, dtype = dtype)  # well, it was a tensor even before, but now we know its properties

    spectr_intens_target = u.fourier(temp_intens_target)
    spectr_intens_target = u.cut(spectr_intens_target, param.freq_width/param.init_freq_res) # we leave central 100 GHz, we delete the rest in order to save GPU
    spectr_intens_target = u.increase_resolution(spectr_intens_target, param.increase_freq_res, device = device, dtype = dtype) # now it's comp_resolution               

    # generate phases
    temp_phase_pred, spectr_phase_pred = model(target_pulse)

    # into frequency domain

    initial_intensity_pt = u.np_to_complex_pt(initial_pulse.Y, device = device, dtype = dtype)
    temp_intens_pred = initial_intensity_pt.clone()
    spectr_intens_pred = u.fourier(initial_intensity_pt)
    # we apply spectral phase

    old_length = np.array(spectr_intens_pred.shape)[-1]
    spectr_intens_pred = u.cut(spectr_intens_pred, param.freq_width/param.init_freq_res) # we leave central 100 GHz, we delete the rest in order to save GPU
    new_length = np.array(spectr_intens_pred.shape)[-1]
    increase_time_res = old_length/new_length

    spectr_intens_pred = u.increase_resolution(spectr_intens_pred, param.increase_freq_res, device = device, dtype = dtype)
    spectr_phase_pred = u.increase_resolution(spectr_phase_pred, param.pulse_shaper_res/param.comp_freq_res, device = device, dtype = dtype)  # 1.5 GHz is the resolution of the pulse shaper
    spectr_idx_start = floor((spectr_intens_pred.shape[-1]-spectr_phase_pred.shape[-1])/2)
    spectr_intens_pred = u.multiply_by_phase(spectr_intens_pred, spectr_phase_pred, index_start = spectr_idx_start, device = device, dtype = dtype)
    spectr_X = np.array([-param.freq_width*1000/2 + param.comp_freq_res*1000*n for n in range(spectr_intens_pred.shape[-1])])# this is in GHz!!!

    # and back to time domain
    temp_intens_pred2 = u.inv_fourier(spectr_intens_pred)
    temp_intens_pred2 = u.increase_resolution(temp_intens_pred2, increase_time_res, device = device, dtype = dtype)
    temp_intens_pred2 = u.cut(temp_intens_pred2, np.array(len(target_pulse)))

    # we apply temporal phase
    temp_phase_pred = u.increase_resolution(temp_phase_pred, param.eopm_res/param.comp_time_res, device = device, dtype = dtype) # 11 ps is the resolution of EOPM
    temp_intens_pred2 = u.multiply_by_phase(temp_intens_pred2, temp_phase_pred, index_start = param.temp_idx_start, device = device, dtype = dtype)

    # create plots

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    # plot 1
    ax1.plot(initial_pulse.X, np.abs(initial_pulse.Y)**2, color="red", zorder = 10, lw = 2)     
    ax1.plot(initial_pulse.X, (temp_intens_target.clone().detach().cpu().numpy().flatten())**2, color = "blue", alpha = 0.5, lw =5, zorder = 0)            
    ax1.set_title("Step 1")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Normalized intensity")
    ax1.set_xlim([-1000, 500])
    ax1.grid()
    
    ax1.legend(["Initial signal", "Target signal"], 
                    facecolor="white", framealpha=1, loc="upper right")

    # legend and phase for ax1
    
    '''
    ax1_ph = plt.twinx(ax1) # ax1 for the phase
    temp_idx_end = np.array(temp_phase_pred.shape)[-1] + param.temp_idx_start   # we want to find the indices of the interval in
    
    x = [initial_pulse.X[param.temp_idx_start: temp_idx_end][0]]
    y = [np.real(temp_phase_pred.clone().detach().cpu().numpy())[0]]
    ax1_ph.plot(x, y, color="red", zorder = 10, lw = 2)     
    ax1_ph.plot(x, y, color = "blue", alpha = 0.5, lw =5, zorder = 0)    
    ax1_ph.plot(initial_pulse.X[param.temp_idx_start: temp_idx_end],
                 np.unwrap(np.real(temp_phase_pred.clone().detach().cpu().numpy())),
                   linestyle = "dashed", color = "darkorange", zorder = 1)
    
    ax1_ph.legend(["Initial signal", "Target signal", "Temporal phase in EOPM"], 
                        facecolor="white", framealpha=1, loc="upper right")
    '''

    # plot 2
    xlim = [-15, 15]

    ax2.plot(spectr_X, np.abs(spectr_intens_pred.clone().detach().cpu().numpy().flatten())**2, color="red", zorder = 10, lw =2)
    ax2.plot(spectr_X, np.abs(spectr_intens_target.clone().detach().cpu().numpy().flatten())**2, color = "darkorange", alpha = 0.7, lw = 5, zorder = 0)
    ax2.set_title("Step 2")
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
    ax2_ph.legend(["Transformed signal", "Target signal", "Spectral phase in P-Sh"],
                                          facecolor="white", framealpha=1, loc="upper right")

    # plot 3

    ax3.plot(initial_pulse.X, np.abs(temp_intens_target.clone().detach().cpu().numpy().flatten())**2, color = "blue", alpha = 0.5, lw =5, zorder = 0)            
    ax3.plot(initial_pulse.X, np.abs(temp_intens_pred2.clone().detach().cpu().numpy().flatten())**2, color="red", lw = 2)    
    ax3.set_title("Step 3")
    ax3.set_xlabel("Time (ps)")
    ax3.set_ylabel("Normalized intensity")
    ax3.grid()
    ax3.set_xlim([-1500, 1500])

    # phase of ax3 and legend

    ax3_ph = plt.twinx(ax3)
    idx_sp_ph_start = np.searchsorted(initial_pulse.X, -1200)
    idx_sp_ph_end = np.searchsorted(initial_pulse.X, 1000)

    x = [initial_pulse.X[idx_sp_ph_start:idx_sp_ph_end][0]]
    y = [np.angle(temp_intens_pred2.clone().detach().cpu().numpy().flatten())[idx_sp_ph_start:idx_sp_ph_end][0]]

    ax3_ph.plot(x, y, color = "red", lw = 2) 
    ax3_ph.plot(x, y, color = "blue", alpha = 0.5, lw = 5, zorder = 0)            
    
    temp_idx_end = np.array(temp_phase_pred.shape)[-1] + param.temp_idx_start   # we want to find the indices of the interval in
     
    ax3_ph.plot(initial_pulse.X[param.temp_idx_start: temp_idx_end],
                 np.unwrap(np.real(temp_phase_pred.clone().detach().cpu().numpy())),
                   linestyle = "dashed", color = "darkorange", zorder = 1)
    
    ax3_ph.plot(initial_pulse.X[idx_sp_ph_start: idx_sp_ph_end], # Residual phase
                np.angle(temp_intens_pred2.clone().detach().cpu().numpy().flatten())[idx_sp_ph_start: idx_sp_ph_end], 
                color = "blue", alpha = 1, linestyle = "dashed")  

    ax3_ph.legend(["Transformed signal", "Target signal", "Temporal phase in EOPM", "Residual temporal phase"],
                                        facecolor="white", framealpha=1, loc="upper right")      
    
    # statistics

    t_target = temp_intens_target.clone().detach().cpu().numpy().flatten()
    t_pred = temp_intens_pred2.clone().detach().cpu().numpy().flatten()
    s_target = spectr_intens_target.clone().detach().cpu().numpy().flatten()
    s_pred = spectr_intens_pred.clone().detach().cpu().numpy().flatten()
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

    init_hom_value =  np.sum(initial*np.conjugate(t_target))*np.sum(np.conjugate(initial)*t_target)
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