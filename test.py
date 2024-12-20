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

def reverse_transformation(model, test_pulse, initial_pulse, device, dtype, save, test_phase = None, iter_num = 0, x_type = "freq"):
    mse = MSELoss()

    input_dim = model.input
    output_dim = model.output
    spectrum_len = len(initial_pulse)
    zeros_num = floor((spectrum_len - input_dim)/2)

    initial_pulse_short = initial_pulse.cut(start = zeros_num, end = zeros_num+input_dim, inplace = False, how = "index")

    plot_from = floor(0*input_dim)
    plot_to = floor(1*input_dim)

    # generate test chirp pulse

    test_phase_pred = model(test_pulse.abs())
    test_phase_pred = test_phase_pred.reshape([output_dim]) #spectral phase of transformation

    # evolve

    initial_intensity = np_to_complex_pt(np.abs(initial_pulse.Y.copy()), device = device, dtype = dtype)
    test_intensity = evolve_pt(initial_intensity, test_phase_pred, device = device, dtype = dtype, abs = False)
    reconstructed = test_intensity.abs()[:, zeros_num: zeros_num+input_dim]
    temporal_phase = torch.angle(test_intensity)[:, zeros_num: zeros_num+input_dim] #of reconstructed signal
    # reverse
    test_pulse_complex = test_pulse.clone()
    test_pulse_temporal = torch.mul(test_pulse_complex, torch.exp(1j*temporal_phase))
    test_pulse_reversed = evolve_pt(test_pulse_temporal, -test_phase_pred, device = device, dtype = dtype, abs = False)
    test_pulse_reversed_detached=test_pulse_reversed.detach().numpy()
    
    
    # create plots

    plt.figure(figsize = (10, 5), constrained_layout=True )

    plt.subplot(1, 2, 1)
    plt.title("Time domain")

    # constant to normalize the time plot

    norm_const = max([np.max(np.abs(initial_pulse_short.Y[plot_from:plot_to])),
                     np.max(np.abs(np.reshape(test_pulse.clone().cpu().detach().numpy(), input_dim))[plot_from:plot_to]),
                     np.max(np.abs(np.reshape(reconstructed.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to]))])
   
    # initial intensity
    plt.plot(initial_pulse_short.X[plot_from:plot_to], 
                np.abs(initial_pulse_short.Y[plot_from:plot_to])/norm_const, 
                color = "blue", 
                zorder = 5)
    
    # target intensity
    plt.plot(initial_pulse_short.X[plot_from:plot_to], 
                    np.abs(np.reshape(test_pulse.clone().cpu().detach().numpy(), input_dim))[plot_from:plot_to]/norm_const, 
                    color = "red")
    
    # transformed intensity
    plt.scatter(initial_pulse_short.X[plot_from:plot_to], 
            np.abs(np.reshape(reconstructed.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to])/norm_const, 
            color = "green", 
            s = 0.25,
            zorder = 10)
    
    #reversed intensity
    plt.plot(initial_pulse_short.X[plot_from:plot_to], 
                np.abs(test_pulse_reversed_detached[plot_from:plot_to]).reshape(test_pulse_reversed.shape[1],)/norm_const, 
                color = "black", 
                zorder = 5)

    
    plt.xlabel("Time (ps)")
    plt.ylabel("Normalized intensity")
    plt.legend(["Initial intensity", "Target intensity", "Transformed intensity", "Reverse transformation"], bbox_to_anchor = [1, -0.12], ncol = 2)
    plt.grid()

    # temporal phase

    ax = plt.gca()
    ax2 = ax.twinx()

    ax2.scatter(initial_pulse_short.X[plot_from:plot_to], 
            np.unwrap(np.reshape(temporal_phase.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to]), 
            color = "burlywood",
            s = 0.25,
            zorder = 0)
    
    ax2.legend(["Phase of transformed spectrum"], bbox_to_anchor = [0.721, -0.25])
    ax2.set_ylabel("Temporal phase (rad)")
    
    # second plot in frequency

    plt.subplot(1, 2, 2)
    
    if x_type == "freq":
        plt.title("Frequency domain")
        plt.xlabel("Frequency (THz)")
    elif x_type == "wl":
        plt.title("Wavelength domain")
        plt.xlabel("Wavelength (nm)")
    else:
        raise Exception("x_type must be either \"wl\" or \"freq\"")
    
    plt.ylabel("Normalized intensity")
    plt.grid()

    # preprocessing

    reconstructed_phase = np.unwrap(test_phase_pred.clone().cpu().detach().numpy().reshape(output_dim))
    reconstructed_phase -= reconstructed_phase[floor(output_dim/2)]

    idx_start = floor(zeros_num + input_dim/2 - output_dim/2)
    idx_end = floor(zeros_num + input_dim/2 + output_dim/2)

    FT_pulse = initial_pulse.inv_fourier(inplace = False)
    FT_Y = FT_pulse.Y.copy()
    FT_X = FT_pulse.X.copy()

    FT_Y /= np.max(FT_Y[idx_start: idx_end])

    # FT intensity

    if x_type == "freq":
        plt.fill_between(FT_X[idx_start: idx_end] + 375, 
                            np.abs(FT_Y[idx_start: idx_end]),
                            color='orange',
                            alpha = 0.5)
        
    elif x_type == "wl":
        plt.fill_between(freq_to_wl(FT_X[idx_start: idx_end] + 375), 
                    np.flip(np.abs(FT_Y[idx_start: idx_end])),
                    color='orange',
                    alpha = 0.5)
        
    else:
        raise Exception("x_type must be either \"wl\" or \"freq\"")


    plt.legend(["FT initial intensity"], bbox_to_anchor = [0.665, -0.12])

    # transforming phase

    ax3 = plt.gca()
    ax4 = ax3.twinx()

    if x_type == "freq":
        ax4.scatter(FT_X[idx_start: idx_end] + 375, 
                    reconstructed_phase, 
                    s = 1, 
                    color = "firebrick",
                    zorder = 10)
        
    elif x_type == "wl":
        ax4.scatter(wl_to_freq(FT_X[idx_start: idx_end] + 375), 
                    np.flip(reconstructed_phase), 
                    s = 1, 
                    color = "firebrick",
                    zorder = 10)
        
    else:
        raise Exception("x_type must be either \"wl\" or \"freq\"")

    ax4.set_ylabel("Spectral phase (rad)")
    ax4.legend(["Transforming phase (rad)"], bbox_to_anchor = [0.8, -0.19])

    # the below part of the code isn't always executed and when is, won't probably work correctly

    if type(test_phase) == type(np.array([])):
        test_phase_np = test_phase.copy()
        #test_phase_np -= test_phase_np[floor(output_dim/2)]
        plt.plot(FT_X[idx_start: idx_end] + 375,
                    np.real(test_phase_np),
                    color = "black",
                    lw = 1,
                    linestyle = "dashed",
                    zorder = 5)
    
        
    '''
    if type(test_phase) == type(np.array([])):
        plt.legend(["Reconstructed phase", "Initial phase", "FT intensity"], bbox_to_anchor = [0.95, -0.15])
    else:
        plt.legend(["Reconstructed phase", "FT intensity"], bbox_to_anchor = [0.95, -0.15])
    '''
    
    if save:
        if not os.path.isdir("pics"):
            os.mkdir("pics")
        plt.savefig("pics/reverse_transformation{}.jpg".format(iter_num), bbox_inches = "tight", dpi = 200)

    return plt, mse(test_pulse.abs(), reconstructed.abs()).clone().cpu().detach().numpy()
    
def test(model, 
         target_pulse, 
         initial_pulse, 
         device, 
         dtype, 
         save, 
         iter_num = 0, 
         x_type = "freq",
         filter_threshold = 1):
    '''
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

    mse = MSELoss()

    input_dim = model.input
    output_dim = model.output
    spectrum_len = len(initial_pulse)
    zeros_num = floor((spectrum_len - input_dim)/2)

    initial_pulse_short = initial_pulse.cut(start = zeros_num, end = zeros_num+input_dim, inplace = False, how = "index")

    plot_from = floor(0*input_dim)
    plot_to = floor(1*input_dim)-1

    # generate test chirp pulse

    target_phase_pred = model(target_pulse.abs())
    target_phase_pred = target_phase_pred.reshape([output_dim])

    filter_mask = u.gen_filter_mask(threshold = filter_threshold, num = len(target_phase_pred), device = device)
    target_phase_pred = u.low_pass_pt(target_phase_pred, filter_mask)

    # evolve

    initial_intensity = np_to_complex_pt(np.abs(initial_pulse.Y.copy()), device = device, dtype = dtype)

    target_intensity_pred = evolve_pt(initial_intensity, target_phase_pred, device = device, dtype = dtype, abs = False)
    reconstructed = target_intensity_pred.abs()[:, zeros_num: zeros_num+input_dim]
    temporal_phase = torch.angle(target_intensity_pred)[:, zeros_num: zeros_num+input_dim]
    
    # create plots

    plt.figure(figsize = (10, 5), constrained_layout = True)

    plt.subplot(1, 2, 1)
    plt.title("Time domain")

    # just for the legend

    x_far_away = 2*initial_pulse_short.X[plot_to]
    plt.plot([x_far_away],[0], color = "blue", lw = 2)
    plt.plot([x_far_away],[0], color = "red", lw = 2)
    plt.plot([x_far_away],[0], color = "green", lw = 6, alpha = 0.5)
    #plt.plot([x_far_away],[0], color = "skyblue")   
    plt.plot([x_far_away],[0], color = "lightcoral", lw = 2, linestyle = "dashed")
    plt.legend(["Initial intensity", "Transformed intensity", "Target intensity", "Phase of transformed spectrum"], 
               bbox_to_anchor = [1.2, -0.12], ncol = 2)

    # constant to normalize the time plot
    norm_const = max([np.max(np.abs(initial_pulse_short.Y[plot_from:plot_to])),
                     np.max(np.abs(np.reshape(target_pulse.clone().cpu().detach().numpy(), input_dim))[plot_from:plot_to]),
                     np.max(np.abs(np.reshape(reconstructed.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to]))])
   
    # initial intensity
    plt.plot(initial_pulse_short.X[plot_from:plot_to], 
                np.abs(initial_pulse_short.Y[plot_from:plot_to])/norm_const, 
                color = "blue", 
                zorder = 5,
                lw = 2)
    
    # target intensity
    plt.plot(initial_pulse_short.X[plot_from:plot_to], 
                    np.abs(np.reshape(target_pulse.clone().cpu().detach().numpy(), input_dim))[plot_from:plot_to]/norm_const, 
                    color = "green",
                    lw = 6,
                    alpha = 0.5)
    
    # transformed intensity
    plt.scatter(initial_pulse_short.X[plot_from:plot_to], 
            np.abs(np.reshape(reconstructed.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to])/norm_const, 
            color = "red", 
            s = 0.25,
            zorder = 10)
    
    plt.xlabel("Time (ps)")
    plt.ylabel("Normalized intensity")
    #plt.legend(["Initial intensity", "Target intensity", "Transformed intensity"], bbox_to_anchor = [1, -0.12], ncol = 2)
    plt.grid()
    plt.xlim([initial_pulse_short.X[plot_from], initial_pulse_short.X[plot_to]])

    # temporal phase, firstly we want to find non-zero intensity

    left = initial_pulse_short.quantile(0.02, norm = "L1") 
    right = initial_pulse_short.quantile(0.98, norm = "L1")
    left_idx = np.searchsorted(initial_pulse_short.X, left)
    right_idx = np.searchsorted(initial_pulse_short.X, right)

    ax = plt.gca()
    ax2 = ax.twinx()

    # initial temporal phase
    '''
    ax2.plot(initial_pulse_short.X[left_idx: right_idx], 
            np.unwrap((np.angle(initial_pulse_short.Y[left_idx: right_idx]))), 
            color = "skyblue",
            lw = 1,
            zorder = 0)
    '''
    # temporal phase

    reconstr_spectrum = sa.spectrum(initial_pulse_short.X[plot_from:plot_to], np.abs(np.reshape(reconstructed.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to])/norm_const, "time", "intensity")
    
    left_2 = reconstr_spectrum.quantile(0.02, norm = "L1") 
    right_2 = reconstr_spectrum.quantile(0.98, norm = "L1")
    left_idx_2 = np.searchsorted(reconstr_spectrum.X, left_2)
    right_idx_2 = np.searchsorted(reconstr_spectrum.X, right_2)

    ax2.plot(initial_pulse_short.X[left_idx_2: right_idx_2], 
            np.unwrap(np.reshape(temporal_phase.clone().cpu().detach().numpy(), input_dim)[left_idx_2:right_idx_2]), 
            color = "lightcoral",
            lw = 2,
            zorder = 0,
            linestyle = "dashed")
    
    #ax2.legend(["Phase of transformed spectrum"], bbox_to_anchor = [0.721, -0.25])
    ax2.set_ylabel("Temporal phase (rad)")
    
    # second plot in frequency

    plt.subplot(1, 2, 2)
    
    if x_type == "freq":
        plt.title("Frequency domain")
        plt.xlabel("Frequency (THz)")
    elif x_type == "wl":
        plt.title("Wavelength domain")
        plt.xlabel("Wavelength (nm)")
    else:
        raise Exception("x_type must be either \"wl\" or \"freq\"")
    
    plt.ylabel("Normalized intensity")
    plt.grid()

    # just for the legend

    plt.fill_between([500], 
                     [0],
                     color = 'orange')
    plt.plot([500], 
             [0],
             lw = 2, 
             color = "firebrick",
             zorder = 10)
    plt.legend(["FT initial intensity", "Transforming phase"], bbox_to_anchor = [0.665, -0.12])

    # preprocessing

    reconstructed_phase = np.unwrap(target_phase_pred.clone().cpu().detach().numpy().reshape(output_dim))
    reconstructed_phase -= reconstructed_phase[floor(output_dim/2)]
    
    reconstructed_phase = u.low_pass_np(reconstructed_phase, filter_mask)

    idx_start = floor(zeros_num + input_dim/2 - output_dim/2)
    idx_end = floor(zeros_num + input_dim/2 + output_dim/2)

    FT_X = initial_pulse.inv_fourier(inplace = False).X
    FT_Y = ifftshift(ifft(ifftshift(torch.flatten(initial_intensity))))
    FT_Y = FT_Y.clone().detach().cpu().numpy()

    FT_Y /= np.max(FT_Y[idx_start: idx_end])

    if x_type == "freq":
        plt.xlim([FT_X[idx_start] + 375, FT_X[idx_end] + 375])
    if x_type == "wl":
        plt.xlim([freq_to_wl(FT_X[idx_end] + 375), freq_to_wl(FT_X[idx_start] + 375)])

    # FT intensity

    if x_type == "freq":
        plt.fill_between(FT_X[idx_start: idx_end] + 375, 
                            np.abs(FT_Y[idx_start: idx_end]),
                            color='orange')
        
    elif x_type == "wl":
        plt.fill_between(freq_to_wl(FT_X[idx_start: idx_end] + 375), 
                    np.flip(np.abs(FT_Y[idx_start: idx_end])),
                    color='orange')
        
    else:
        raise Exception("x_type must be either \"wl\" or \"freq\"")

    # transforming phase

    ax3 = plt.gca()
    ax4 = ax3.twinx()

    if x_type == "freq":
        ax4.plot(FT_X[idx_start: idx_end] + 375, 
                    reconstructed_phase, 
                    lw = 2, 
                    color = "firebrick",
                    zorder = 10)
        
    elif x_type == "wl":
        ax4.plot(wl_to_freq(FT_X[idx_start: idx_end] + 375), 
                    np.flip(reconstructed_phase), 
                    lw = 2, 
                    color = "firebrick",
                    zorder = 10)
        
    else:
        raise Exception("x_type must be either \"wl\" or \"freq\"")

    ax4.set_ylabel("Spectral phase (rad)")

    # the below part of the code isn't always executed and when is, won't probably work correctly
    
    if save:
        if not os.path.isdir("pics"):
            os.mkdir("pics")
        plt.savefig("pics/reconstructed_{}.svg".format(iter_num), bbox_inches = "tight", dpi = 200)

    return plt, mse(target_pulse.abs(), reconstructed.abs()).clone().cpu().detach().numpy()
    

def create_target_pulse(pulse_type, initial_pulse, phase_len, device, dtype):
    '''
    ## Create a target_intensity within given rules.
    # Arguments:

    pulse_type - if \"hermite\", then the target intensity is a 1 Hermite-Gauss polynomial.
    If \"chirp\", then the target intensity is chirped Gaussian function. 
    If \"from_dataset\", then chooses at random a intensity saved in \"data/train_intensity\".
    If \"two_pulses\", then returns two separated gaussian pulses.

    initial_pulse - a spectrum class object containing the initial spectrum that is - possibly - transformed into target_pulse.

    phase_len - the length of significant part of the Fourier transformed initial_pulse

    # Returns:
    "target_pulse" being one-dimensional complex PyTorch Tensor.
    '''

    if pulse_type == "hermite":
        target_pulse_ = sa.hermitian_pulse(pol_num = 1,
                                        bandwidth = (initial_pulse.X[0], initial_pulse.X[-1]),
                                        centre = 500,
                                        FWHM = 100,
                                        num = len(initial_pulse),
                                    x_type = "time")

        target_pulse_.Y = target_pulse_.Y / np.sqrt(np.sum(target_pulse_.Y*np.conjugate(target_pulse_.Y)))
        target_pulse_.Y = target_pulse_.Y * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))
        
        target_pulse_.very_smart_shift(target_pulse_.comp_center(norm = "L2")-initial_pulse.comp_center(norm = "L2"))
        target_pulse_ = np_to_complex_pt(target_pulse_.Y, device = device, dtype = dtype)

    elif pulse_type == "chirp":
        if dtype == torch.float32:
            new_dtype = np.float32
        else:
            new_dtype = dtype
        initial_intensity = initial_pulse.Y.copy()
        chirp = 100
        transform_phase = chirp*np.linspace(-1, 1, phase_len, dtype = new_dtype)**2
        target_pulse_ = evolve_np(initial_intensity, transform_phase, dtype = new_dtype)

        target_pulse_ = shift_to_centre(target_pulse_, initial_pulse.Y)
        target_pulse_ = np_to_complex_pt(target_pulse_, device = device, dtype = torch.float32)

    elif pulse_type == "two_pulses":
        pulses = sa.hermitian_pulse(pol_num = 0, 
                                    bandwidth = [initial_pulse.X[0], initial_pulse.X[-1]],
                                    centre = initial_pulse.quantile(0.5),
                                    FWHM = initial_pulse.FWHM(),
                                    num = len(initial_pulse),
                                    x_type = initial_pulse.x_type)
        pulses.Y = pulses.Y + pulses.very_smart_shift(-0.5, inplace = False).Y + pulses.very_smart_shift(0.5, inplace = False).Y
        pulses.Y = pulses.Y / np.sqrt(np.sum(pulses.Y*np.conjugate(pulses.Y)))
        pulses.Y = pulses.Y * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))

        target_pulse_.very_smart_shift(target_pulse_.comp_center()-initial_pulse.comp_center())
        target_pulse_ = np_to_complex_pt(pulses.Y, device = device, dtype = dtype)

    elif pulse_type == "from_dataset":
        intensity_labels = os.listdir('data/train_intensity')
        phase_labels = os.listdir('data/train_phase')
        dataset_size = len(intensity_labels)
        number = np.random.randint(low = 0, high = dataset_size)
        intensity_name = intensity_labels[number]
        phase_name = phase_labels[number]

        target_pulse_ = np.loadtxt('data/train_intensity/' + intensity_name,
                 delimiter = " ", dtype = np.float32)
        transform_phase = np.loadtxt('data/train_phase/' + phase_name,
                 delimiter = " ", dtype = np.float32)
        
        target_pulse_ = shift_to_centre(target_pulse_, initial_pulse.Y)
        target_pulse_ = np_to_complex_pt(target_pulse_, device = device, dtype = dtype)

    elif pulse_type == "exponential":
        exp_intensity = np.flip(np.exp(np.linspace(-3, 3, len(initial_pulse))) - np.exp(-3))
        for i in range(0, floor(len(exp_intensity)*1/3)):
            exp_intensity[i] = 0
        exp_intensity = exp_intensity / np.sqrt(np.sum(exp_intensity*np.conjugate(exp_intensity)))
        exp_intensity = exp_intensity * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))

        #exp_intensity = shift_to_centre(exp_intensity, initial_pulse.Y)
        target_pulse_ = np_to_complex_pt(exp_intensity, device = device, dtype = dtype)

    elif pulse_type == "gauss":
        target_pulse_ = sa.hermitian_pulse(pol_num = 0,
                                    bandwidth = (initial_pulse.X[0], initial_pulse.X[-1]),
                                    centre = 500,
                                    FWHM = 200,
                                    num = len(initial_pulse),
                                    x_type = "time")

        target_pulse_.Y = target_pulse_.Y / np.sqrt(np.sum(target_pulse_.Y*np.conjugate(target_pulse_.Y)))
        target_pulse_.Y = target_pulse_.Y * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))

        target_pulse_.smart_shift(-target_pulse_.comp_center(norm = "L2")+initial_pulse.comp_center(norm = "L2"))
        target_pulse_ = np_to_complex_pt(target_pulse_.Y, device = device, dtype = dtype)

    else:
        raise Exception("Pulse_type not defined.")

    return target_pulse_.clone()


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
        Y = np.flip(np.exp(np.linspace(-10, 3, num)) - np.exp(-10))
        for i in range(0, floor(1/3*num)):
            Y[i] = 0

        X = np.linspace(bandwidth[0], bandwidth[1], num)
        spectrum_out = sa.spectrum(X = X, Y = Y, x_type ="time", y_type ="intensity")
        spectrum_out.smart_shift(100, inplace = True)
        spectrum_out.Y = np.abs(spectrum_out.Y)
        return spectrum_out
    
    else:
        raise Exception("Pulse_type must be either \"gauss\", \"hermite\" or \"exponential\".")
    


def create_test_set(initial_pulse, phase_len, device, dtype):
    '''
    ## Returns a list with predefined test intensities.

    initial_pulse - a spectrum class object containing the initial spectrum that is - possibly - transformed into test_pulse.

    phase_len - the length of significant part of the Fourier transformed initial_pulse
    '''
    test_set = []
    for pulse_type in ["hermite", "chirp", "exponential", "gauss"]:
        test_set.append(create_test_pulse(pulse_type = pulse_type, 
                                          initial_pulse = initial_pulse.copy(),
                                          phase_len = phase_len, 
                                          device = device, 
                                          dtype = dtype)[0])
        
    return test_set