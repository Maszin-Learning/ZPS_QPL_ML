import numpy as np
import matplotlib.pyplot as plt
from math import floor
import os
import spectral_analysis as sa
from utilities import np_to_complex_pt, evolve_np, evolve_pt

def test(model, test_pulse, initial_pulse_Y, initial_pulse_X, device, dtype, test_phase = None, iter_num = 0):
    '''
    You may provide test phase IF you know it. Otherwise leave it to be None.
    '''

    def mse(actual_data, predictions):
        return np.mean((actual_data - predictions)**2)

    input_dim = model.input
    output_dim = model.output

    plot_from = floor(input_dim*1/6)
    plot_to = floor(input_dim*5/6)

    # generate test chirp pulse

    test_phase_pred = model(test_pulse.abs())
    test_phase_pred = test_phase_pred.reshape([output_dim])

    # evolve

    initial_intensity = np_to_complex_pt(initial_pulse_Y.copy(), device = device, dtype = dtype)

    test_intensity = evolve_pt(initial_intensity, test_phase_pred, device = device, dtype = dtype)
    reconstructed = test_intensity.abs() 

    plt.subplot(1, 2, 1)
    plt.title("The intensity")

    plt.scatter(initial_pulse_X[plot_from:plot_to], 
                np.abs(np.reshape(reconstructed.clone().cpu().detach().numpy(), input_dim)[plot_from:plot_to]), 
                color = "green", 
                s = 1,
                zorder = 10)
    plt.plot(initial_pulse_X[plot_from:plot_to], 
                np.abs(initial_pulse_Y[plot_from:plot_to]), 
                linestyle = "dashed", 
                color = "black", 
                lw = 1,
                zorder = 5)
    plt.plot(initial_pulse_X[plot_from:plot_to], 
                        np.abs(np.reshape(test_pulse.clone().cpu().detach().numpy(), input_dim))[plot_from:plot_to], 
                        color = "darkviolet")
    plt.xlabel("THz")
    plt.legend(["Reconstructed intensity", "Initial intensity", "Target intensity"], bbox_to_anchor = [0.95, -0.15])
    plt.grid()

    plt.subplot(1, 2, 2)

    plt.title("The phase")

    reconstructed_phase = np.unwrap(test_phase_pred.clone().cpu().detach().numpy().reshape(output_dim))
    reconstructed_phase -= reconstructed_phase[floor(output_dim/2)]

    idx_start = floor(input_dim/2 - output_dim/2)
    idx_end = floor(input_dim/2 + output_dim/2)

    FT_intensity = initial_pulse_Y.copy()
    FT_intensity = np.fft.fftshift(FT_intensity)
    FT_intensity = np.fft.fft(FT_intensity)
    FT_intensity = np.fft.fftshift(FT_intensity)

    plt.scatter(range(idx_end - idx_start), 
                reconstructed_phase, 
                s = 1, 
                color = "red",
                zorder = 10)

    if test_phase != None:
        test_phase_np = test_phase.clone().cpu().detach().numpy()
        test_phase_np -= test_phase_np[floor(output_dim/2)]
        plt.plot(range(idx_end - idx_start),
                    np.real(test_phase_np),
                    color = "black",
                    lw = 1,
                    linestyle = "dashed",
                    zorder = 5)

    FT_intensity /= np.max(FT_intensity[idx_start: idx_end])
    if test_phase != None:
        FT_intensity *= np.max(np.concatenate([np.abs(reconstructed_phase), np.abs(test_phase.clone().detach().cpu().numpy())]))
    else:
        FT_intensity *= np.max(np.abs(reconstructed_phase))

    plt.plot(range(idx_end - idx_start), 
                        np.abs(FT_intensity[idx_start: idx_end]),
                        color='red')
    
    plt.xlabel("Quasi-time (unitless)")
    if test_phase != None:
        plt.legend(["Reconstructed phase", "Initial phase", "FT intensity"], bbox_to_anchor = [0.95, -0.15])
    else:
        plt.legend(["Reconstructed phase", "FT intensity"], bbox_to_anchor = [0.95, -0.15])
    plt.grid()

    if not os.path.isdir("pics"):
        os.mkdir("pics")

    loss = mse(np.abs(np.reshape(test_pulse.clone().cpu().detach().numpy(), input_dim)), reconstructed)

    plt.savefig("pics/reconstructed_0{}.jpg".format(iter_num), bbox_inches = "tight", dpi = 200)
    return plt, loss
    

def create_test_pulse(pulse_type, initial_pulse, phase_len, device, dtype):
    '''
    pulse_type for now must be either \"hermite\", \"chirp\" or \"random_evolution\".
    A tuple (test_pulse, test_phase) is returned where test_phase is the phase used to evolve the gaussian
    to the test_pulse. If the pulse wasn't obtained by such evolution, then it is equal to None.
    '''

    if pulse_type == "hermite":
        test_pulse = sa.hermitian_pulse(pol_num = 1,
                                        bandwidth = (initial_pulse.X[0], initial_pulse.X[-1]),
                                        centre = 193,
                                        FWHM = 1,
                                        num = len(initial_pulse))

        test_pulse.Y /= np.sum(test_pulse.Y*np.conjugate(test_pulse.Y))
        test_pulse.Y *= np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y))
        
        test_pulse = np_to_complex_pt(test_pulse.Y, device = device, dtype = dtype)
        test_phase = None

    elif pulse_type == "chirp":
        test_pulse = initial_pulse.copy()
        chirp = 20
        test_phase = np_to_complex_pt(chirp*np.linspace(-1, 1, phase_len)**2, device = device, dtype = dtype)
        test_phase = test_phase.reshape([phase_len])
        test_pulse = evolve_np(np_to_complex_pt(test_pulse.Y, device = device, dtype = dtype), test_phase, device = device, dtype = dtype)

    elif pulse_type == "translation":
        test_phase = 5*np.linspace(-1, 1, phase_len)
        test_pulse = evolve_np(initial_pulse.Y, test_phase)

    return test_pulse, test_phase