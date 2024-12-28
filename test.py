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
         comp_time_resolution,
         temp_idx_start,
         init_freq_resolution,
         increase_freq_res,
         comp_freq_resolution,
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

    mse = MSELoss()

    # generate test chirp pulse

    temp_phase_pred, spectr_phase_pred = model(target_pulse)

    # we apply temporal phase
    temp_phase_pred = u.increase_resolution(temp_phase_pred, 11/comp_time_resolution, device = device, dtype = dtype) # 11 ps is the resolution of EOPM
    initial_intensity_pt = u.np_to_complex_pt(initial_pulse.Y, device = device, dtype = dtype)
    temp_intens_pred = u.multiply_by_phase(initial_intensity_pt, temp_phase_pred, index_start = temp_idx_start, device = device, dtype = dtype)

    # we apply spectral phase
    spectr_intens_pred = u.fourier(temp_intens_pred)
    spectr_intens_pred = u.cut(spectr_intens_pred, 50/init_freq_resolution) # we leave central 50 GHz, we delete the rest in order to save GPU
    spectr_intens_pred = u.increase_resolution(spectr_intens_pred, increase_freq_res, device = device, dtype = dtype)
    spectr_phase_pred = u.increase_resolution(spectr_phase_pred, 0.0015/comp_freq_resolution, device = device, dtype = dtype)  # 1.5 GHz is the resolution of the pulse shaper
    spectr_intens_pred = u.multiply_by_phase(spectr_intens_pred, spectr_phase_pred, index_start = floor((spectr_intens_pred.shape[-1]-spectr_phase_pred.shape[-1])/2), device = device, dtype = dtype)

    # and back to time domain
    temp_intens_pred2 = u.inv_fourier(spectr_intens_pred)
    temp_intens_pred2 = u.cut(temp_intens_pred2, np.array(len(target_pulse)))
    
    # create plots

    plt.figure(figsize = (10, 10), constrained_layout = True)

    plt.subplot(2, 2, 1)

    plt.plot(initial_pulse.X, initial_pulse.Y, color = "darkviolet")
    plt.title("Step 1")
    plt.xlabel("Time (ps)")
    plt.ylabel("Normalized intensity")
    plt.grid()

    plt.subplot(2, 2, 2)

    plt.plot(range(spectr_intens_pred.shape[-1]), np.abs(spectr_intens_pred.clone().detach().cpu().numpy().flatten()), color = "darkorange")
    plt.title("Step 2")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Normalized intensity")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(range(temp_intens_pred2.shape[-1]), np.abs(temp_intens_pred2.clone().detach().cpu().numpy().flatten()), color = "darkviolet")
    plt.title("Step 3")
    plt.xlabel("Time (ps)")
    plt.ylabel("Normalized intensity")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.title("Step 4")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Normalized intensity")
    plt.grid()

    if save:
        if not os.path.isdir("pics"):
            os.mkdir("pics")
        plt.savefig("pics/reconstructed_{}.svg".format(iter_num), bbox_inches = "tight", dpi = 200)

    return plt, 0
    

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