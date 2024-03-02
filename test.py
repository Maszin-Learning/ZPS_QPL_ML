import numpy as np
import matplotlib.pyplot as plt
from math import floor
import os
import spectral_analysis as sa
from utilities import np_to_complex_pt, evolve_np, evolve_pt, shift_to_centre, wl_to_freq, freq_to_wl
from torch.nn import MSELoss
import torch
from scipy.interpolate import CubicSpline
from scipy.interpolate import splrep, BSpline

def test(model, test_pulse, initial_pulse, device, dtype, save, test_phase = None, iter_num = 0, x_type = "freq", flag = "", reconst_phase = None):
    '''
    ## Test the model with a given test pulse.

    # Arguments:

    model - the model of the neural network.

    test_pulse - one-dimensional complex Pytorch Tensor.

    initial_pulse - a spectrum class object

    test_phase - one-dimensional real-valued NumPy array or None - it serves only for plotting.

    iter_num - the test plot is saved as \"pics/reconstructed_[iter_num].jpg\"

    # Returns:

    (plot, loss) - where plot (returned in a strange way) depicts model predictions on test pulse and phase, 
    and loss is MSE of that prediction.

    # Note:

    1. initial_pulse_Y, initial_pulse_X and test_pulse must have the same length.

    2. The length of test_phase should be equal to the length of the significant part of Fourier-transformed initial_pulse_Y.
    '''

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
    test_phase_pred = test_phase_pred.reshape([output_dim])

    # evolve

    initial_intensity = np_to_complex_pt(np.abs(initial_pulse.Y.copy()), device = device, dtype = dtype)
    test_intensity = evolve_pt(initial_intensity, test_phase_pred, device = device, dtype = dtype, abs = False)
    reconstructed = test_intensity.abs()[:, zeros_num: zeros_num+input_dim]
    temporal_phase = torch.angle(test_intensity)[:, zeros_num: zeros_num+input_dim]
    
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
    
    plt.xlabel("Time (ps)")
    plt.ylabel("Normalized intensity")
    plt.legend(["Initial intensity", "Target intensity", "Transformed intensity"], bbox_to_anchor = [1, -0.12], ncol = 2)
    plt.grid()

    # temporal phase, firstly we want to find non-zero intensity

    reconstr_spectrum = sa.spectrum(initial_pulse_short.X[plot_from:plot_to], np.abs(np.reshape(reconstructed.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to])/norm_const, "time", "intensity")

    left = reconstr_spectrum.quantile(0.02, norm = "L1") 
    right = reconstr_spectrum.quantile(0.98, norm = "L1")
    left_idx = np.searchsorted(initial_pulse_short.X, left)
    right_idx = np.searchsorted(initial_pulse_short.X, right)

    ax = plt.gca()
    ax2 = ax.twinx()

    ax2.scatter(initial_pulse_short.X[left_idx: right_idx], 
            np.unwrap(np.reshape(temporal_phase.clone().cpu().detach().numpy(), input_dim)[left_idx:right_idx]), 
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

    if reconst_phase != None:
        test_phase_pred = reconst_phase[0, :]

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
        ax4.plot(FT_X[idx_start: idx_end] + 375, 
                    reconstructed_phase, 
                    lw = 1, 
                    color = "red",
                    zorder = 10)
        
    elif x_type == "wl":
        ax4.plot(wl_to_freq(FT_X[idx_start: idx_end] + 375), 
                    np.flip(reconstructed_phase), 
                    lw = 1, 
                    color = "red",
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
        if not os.path.isdir("pics" + flag):
            os.mkdir("pics" + flag)
        plt.savefig("pics" + flag + "/reconstructed_{}.jpg".format(iter_num), bbox_inches = "tight", dpi = 200)

    return plt, mse(test_pulse.abs(), reconstructed.abs()).clone().cpu().detach().numpy()
    

def create_test_pulse(pulse_type, initial_pulse, phase_len, device, dtype):
    '''
    ## Create a test_intensity and test_phase within given rules.
    # Arguments:

    pulse_type - if \"hermite\", then the test intensity is a 1 Hermite-Gauss polynomial.
    If \"chirp\", then the test intensity is chirped Gaussian function. 
    If \"from_dataset\", then chooses at random a intensity saved in \"data/train_intensity\".
    If \"two_pulses\", then returns two separated gaussian pulses.

    initial_pulse - a spectrum class object containing the initial spectrum that is - possibly - transformed into test_pulse.

    phase_len - the length of significant part of the Fourier transformed initial_pulse

    # Returns:
    (test_pulse, test_phase), where test_pulse is one-dimensional complex PyTorch Tensor and test_phase is one-dimensional
     real NumPy Array or None, if the phase needed to transform initial_pulse to the test_pulse is not known.
    '''

    if pulse_type == "hermite_1":
        test_pulse_ = sa.hermitian_pulse(pol_num = 1,
                                        bandwidth = (initial_pulse.X[0], initial_pulse.X[-1]),
                                        centre = 500,
                                        FWHM = 100,
                                        num = len(initial_pulse))

        test_pulse_.Y = test_pulse_.Y / np.sqrt(np.sum(test_pulse_.Y*np.conjugate(test_pulse_.Y)))
        test_pulse_.Y = test_pulse_.Y * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))
        
        test_pulse_.very_smart_shift(test_pulse_.comp_center(norm = "L2")-initial_pulse.comp_center(norm = "L2"))
        test_pulse_ = np_to_complex_pt(test_pulse_.Y, device = device, dtype = dtype)
        test_phase_ = None

    elif pulse_type == "hermite_3":
        test_pulse_ = sa.hermitian_pulse(pol_num = 3,
                                        bandwidth = (initial_pulse.X[0], initial_pulse.X[-1]),
                                        centre = 500,
                                        FWHM = 100,
                                        num = len(initial_pulse))

        test_pulse_.Y = test_pulse_.Y / np.sqrt(np.sum(test_pulse_.Y*np.conjugate(test_pulse_.Y)))
        test_pulse_.Y = test_pulse_.Y * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))
        
        test_pulse_.very_smart_shift(test_pulse_.comp_center(norm = "L2")-initial_pulse.comp_center(norm = "L2"))
        test_pulse_ = np_to_complex_pt(test_pulse_.Y, device = device, dtype = dtype)
        test_phase_ = None

    elif pulse_type == "hermite_2":
        test_pulse_ = sa.hermitian_pulse(pol_num = 2,
                                        bandwidth = (initial_pulse.X[0], initial_pulse.X[-1]),
                                        centre = 500,
                                        FWHM = 100,
                                        num = len(initial_pulse))

        test_pulse_.Y = test_pulse_.Y / np.sqrt(np.sum(test_pulse_.Y*np.conjugate(test_pulse_.Y)))
        test_pulse_.Y = test_pulse_.Y * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))
        
        test_pulse_.very_smart_shift(test_pulse_.comp_center(norm = "L2")-initial_pulse.comp_center(norm = "L2"))
        test_pulse_ = np_to_complex_pt(test_pulse_.Y, device = device, dtype = dtype)
        test_phase_ = None

    elif pulse_type == "chirp":
        if dtype == torch.float32:
            new_dtype = np.float32
        else:
            new_dtype = dtype
        initial_intensity = initial_pulse.Y.copy()
        chirp = 100
        test_phase_ = chirp*np.linspace(-1, 1, phase_len, dtype = new_dtype)**2
        test_pulse_ = evolve_np(initial_intensity, test_phase_, dtype = new_dtype)

        test_pulse_ = shift_to_centre(test_pulse_, initial_pulse.Y)
        test_pulse_ = np_to_complex_pt(test_pulse_, device = device, dtype = torch.float32)

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

        test_pulse_.very_smart_shift(test_pulse_.comp_center()-initial_pulse.comp_center())
        test_pulse_ = np_to_complex_pt(pulses.Y, device = device, dtype = dtype)
        test_phase_ = None

    elif pulse_type == "from_dataset":
        intensity_labels = os.listdir('data/train_intensity')
        phase_labels = os.listdir('data/train_phase')
        dataset_size = len(intensity_labels)
        number = np.random.randint(low = 0, high = dataset_size)
        intensity_name = intensity_labels[number]
        phase_name = phase_labels[number]

        test_pulse_ = np.loadtxt('data/train_intensity/' + intensity_name,
                 delimiter = " ", dtype = np.float32)
        test_phase_ = np.loadtxt('data/train_phase/' + phase_name,
                 delimiter = " ", dtype = np.float32)
        
        test_pulse_ = shift_to_centre(test_pulse_, initial_pulse.Y)
        test_pulse_ = np_to_complex_pt(test_pulse_, device = device, dtype = dtype)

    elif pulse_type == "exponential":
        exp_intensity = np.flip(np.exp(np.linspace(-3, 3, len(initial_pulse))) - np.exp(-3))
        for i in range(0, floor(len(exp_intensity)*1/3)):
            exp_intensity[i] = 0
        exp_intensity = exp_intensity / np.sqrt(np.sum(exp_intensity*np.conjugate(exp_intensity)))
        exp_intensity = exp_intensity * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))

        exp_intensity = shift_to_centre(exp_intensity, initial_pulse.Y)
        test_pulse_ = np_to_complex_pt(exp_intensity, device = device, dtype = dtype)
        test_phase_ = None

    elif pulse_type == "gauss":
        test_pulse_ = sa.hermitian_pulse(pol_num = 0,
                                        bandwidth = (initial_pulse.X[0], initial_pulse.X[-1]),
                                        centre = 500,
                                        FWHM = 100,
                                        num = len(initial_pulse))

        test_pulse_.Y = test_pulse_.Y / np.sqrt(np.sum(test_pulse_.Y*np.conjugate(test_pulse_.Y)))
        test_pulse_.Y = test_pulse_.Y * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))

        test_pulse_.very_smart_shift(test_pulse_.comp_center(norm = "L2")-initial_pulse.comp_center(norm = "L2"))
        test_pulse_ = np_to_complex_pt(test_pulse_.Y, device = device, dtype = dtype)
        test_phase_ = None

    else:
        raise Exception("Pulse_type not defined.")

    return test_pulse_.clone(), test_phase_


def create_initial_pulse(bandwidth, centre, FWHM, num, pulse_type):

    if pulse_type == "gauss":
        pulse = sa.hermitian_pulse(pol_num = 0,
                                    bandwidth = bandwidth,
                                    centre = centre,
                                    FWHM = FWHM,
                                    num = num)
        pulse.Y = np.abs(pulse.Y)
        return pulse
    
    elif pulse_type == "hermite_1":
        pulse = sa.hermitian_pulse(pol_num = 1,
                                    bandwidth = bandwidth,
                                    centre = centre,
                                    FWHM = FWHM,
                                    num = num)
        pulse.Y = np.abs(pulse.Y)
        return pulse
    
    elif pulse_type == "hermite_2":
        pulse = sa.hermitian_pulse(pol_num = 2,
                                    bandwidth = bandwidth,
                                    centre = centre,
                                    FWHM = FWHM,
                                    num = num)
        pulse.Y = np.abs(pulse.Y)
        return pulse
    
    elif pulse_type == "hermite_3":
        pulse = sa.hermitian_pulse(pol_num = 3,
                                    bandwidth = bandwidth,
                                    centre = centre,
                                    FWHM = FWHM,
                                    num = num)
        pulse.Y = np.abs(pulse.Y)
        return pulse
    
    elif pulse_type == "exponential":
        Y = np.flip(np.exp(np.linspace(-10, 3, num)) - np.exp(-10))
        for i in range(0, floor(1/3*num)):
            Y[i] = 0

        X = np.linspace(bandwidth[0], bandwidth[1], num)
        spectrum_out = sa.spectrum(X = X, Y = Y, x_type ="time", y_type ="intensity")
        spectrum_out.very_smart_shift(centre-(bandwidth[1]+bandwidth[0])/2, inplace = True)
        spectrum_out.Y = np.abs(spectrum_out.Y)
        return spectrum_out
    
    elif pulse_type == "exponential_r":
        Y = np.flip(np.exp(np.linspace(-10, 3, num)) - np.exp(-10))
        for i in range(0, floor(1/3*num)):
            Y[i] = 0

        Y = np.flip(Y)
        X = np.linspace(bandwidth[0], bandwidth[1], num)
        spectrum_out = sa.spectrum(X = X, Y = Y, x_type ="time", y_type ="intensity")
        spectrum_out.very_smart_shift(centre-(bandwidth[1]+bandwidth[0])/2, inplace = True)
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