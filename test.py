import numpy as np
import matplotlib.pyplot as plt
from math import floor
import os
import spectral_analysis as sa
from utilities import np_to_complex_pt, evolve_np, evolve_pt, shift_to_centre
from torch.nn import MSELoss
import torch

def test(model, test_pulse, initial_pulse, device, dtype, save, test_phase = None, iter_num = 0, ):
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
    test_intensity = evolve_pt(initial_intensity, test_phase_pred, device = device, dtype = dtype)
    reconstructed = test_intensity.abs()[:,zeros_num : zeros_num+input_dim] 

    plt.subplot(1, 2, 1)
    plt.title("The intensity")

    plt.scatter(initial_pulse_short.X[plot_from:plot_to], 
                np.abs(np.reshape(reconstructed.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to]), 
                color = "green", 
                s = 1,
                zorder = 10)
    plt.plot(initial_pulse_short.X[plot_from:plot_to], 
                np.abs(initial_pulse_short.Y[plot_from:plot_to]), 
                linestyle = "dashed", 
                color = "black", 
                lw = 1,
                zorder = 5)
    plt.plot(initial_pulse_short.X[plot_from:plot_to], 
                        np.abs(np.reshape(test_pulse.clone().cpu().detach().numpy(), input_dim))[plot_from:plot_to], 
                        color = "darkviolet")
    plt.xlabel("THz")
    plt.legend(["Reconstructed intensity", "Initial intensity", "Target intensity"], bbox_to_anchor = [0.95, -0.15])
    plt.grid()

    plt.subplot(1, 2, 2)

    plt.title("The phase")

    reconstructed_phase = np.unwrap(test_phase_pred.clone().cpu().detach().numpy().reshape(output_dim))
    reconstructed_phase -= reconstructed_phase[floor(output_dim/2)]

    idx_start = floor(zeros_num + input_dim/2 - output_dim/2)
    idx_end = floor(zeros_num + input_dim/2 + output_dim/2)

    FT_intensity = initial_pulse.Y.copy()
    FT_intensity = np.fft.fftshift(FT_intensity)
    FT_intensity = np.fft.fft(FT_intensity)
    FT_intensity = np.fft.fftshift(FT_intensity)

    plt.scatter(range(idx_end - idx_start), 
                reconstructed_phase, 
                s = 1, 
                color = "red",
                zorder = 10)

    if type(test_phase) == type(np.array([])):
        test_phase_np = test_phase.copy()
        test_phase_np -= test_phase_np[floor(output_dim/2)]
        plt.plot(range(idx_end - idx_start),
                    np.real(test_phase_np),
                    color = "black",
                    lw = 1,
                    linestyle = "dashed",
                    zorder = 5)

    FT_intensity /= np.max(FT_intensity[idx_start: idx_end])
    if type(test_phase) == type(np.array([])):
        FT_intensity *= np.max(np.concatenate([np.abs(reconstructed_phase), np.abs(test_phase.copy())]))
    else:
        FT_intensity *= np.max(np.abs(reconstructed_phase))

    plt.plot(range(idx_end - idx_start), 
                        np.abs(FT_intensity[idx_start: idx_end]),
                        color='red')
    
    plt.xlabel("Quasi-time (unitless)")
    if type(test_phase) == type(np.array([])):
        plt.legend(["Reconstructed phase", "Initial phase", "FT intensity"], bbox_to_anchor = [0.95, -0.15])
    else:
        plt.legend(["Reconstructed phase", "FT intensity"], bbox_to_anchor = [0.95, -0.15])
    plt.grid()

    if save:
        if not os.path.isdir("pics"):
            os.mkdir("pics")
        plt.savefig("pics/reconstructed_{}.jpg".format(iter_num), bbox_inches = "tight", dpi = 200)

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

    if pulse_type == "hermite":
        test_pulse_ = sa.hermitian_pulse(pol_num = 1,
                                        bandwidth = (initial_pulse.X[0], initial_pulse.X[-1]),
                                        centre = 193,
                                        FWHM = 1,
                                        num = len(initial_pulse))

        test_pulse_.Y = test_pulse_.Y / np.sqrt(np.sum(test_pulse_.Y*np.conjugate(test_pulse_.Y)))
        test_pulse_.Y = test_pulse_.Y * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))
        
        test_pulse_.very_smart_shift(test_pulse_.comp_center()-initial_pulse.comp_center())
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
        exp_intensity = np.exp(np.linspace(-3, 3, len(initial_pulse))) - np.exp(-1.5)
        for i in range(floor(len(exp_intensity)*3/4), len(exp_intensity)):
            exp_intensity[i] = 0
        for i in range(0, floor(len(exp_intensity)*1/4)):
            exp_intensity[i] = 0
        exp_intensity = exp_intensity / np.sqrt(np.sum(exp_intensity*np.conjugate(exp_intensity)))
        exp_intensity = exp_intensity * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))

        exp_intensity = shift_to_centre(exp_intensity, initial_pulse.Y)
        test_pulse_ = np_to_complex_pt(exp_intensity, device = device, dtype = dtype)
        test_phase_ = None

    elif pulse_type == "gauss":
        test_pulse_ = sa.hermitian_pulse(pol_num = 0,
                                    bandwidth = (initial_pulse.X[0], initial_pulse.X[-1]),
                                    centre = 193,
                                    FWHM = 0.5,
                                    num = len(initial_pulse))

        test_pulse_.Y = test_pulse_.Y / np.sqrt(np.sum(test_pulse_.Y*np.conjugate(test_pulse_.Y)))
        test_pulse_.Y = test_pulse_.Y * np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))

        test_pulse_.very_smart_shift(test_pulse_.comp_center()-initial_pulse.comp_center())
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
    
    elif pulse_type == "hermite":
        pulse = sa.hermitian_pulse(pol_num = 1,
                                    bandwidth = bandwidth,
                                    centre = centre,
                                    FWHM = FWHM,
                                    num = num)
        pulse.Y = np.abs(pulse.Y)
        return pulse
    
    elif pulse_type == "exponential":
        Y = np.flip(np.exp(np.linspace(-3, 3, num)) - np.exp(-1.5))
        for i in range(floor(num*3/4), num):
            Y[i] = 0
        for i in range(0, floor(num*1/4)):
            Y[i] = 0
        X = np.linspace(bandwidth[0], bandwidth[1], num)
        spectrum_out = sa.spectrum(X = X, Y = Y, x_type ="freq", y_type ="intensity")
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