def test(model, test_pulse, initial_pulse_Y, initial_pulse_X, test_phase = None):
    '''
    You may provide test phase IF you know it. Otherwise leave it to be None.
    '''
    
    import numpy as np
    from utilities import np_to_complex_pt
    from utilities import evolve
    import matplotlib.pyplot as plt
    from math import floor

    input_dim = model.input
    output_dim = model.output

    plot_from = floor(input_dim*1/6)
    plot_to = floor(input_dim*5/6)

    # generate test chirp pulse

    test_phase_pred = model(test_pulse.abs())
    test_phase_pred = test_phase_pred.reshape([output_dim])

    # evolve

    initial_intensity = np_to_complex_pt(initial_pulse_Y.copy())

    test_intensity = evolve(initial_intensity, test_phase_pred)
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
    plt.fill_between(initial_pulse_X[plot_from:plot_to], 
                        np.abs(np.reshape(test_pulse.clone().cpu().detach().numpy(), input_dim))[plot_from:plot_to], 
                        color = "darkviolet", 
                        alpha = 0.2,
                        zorder = 0)
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

    plt.fill_between(range(idx_end - idx_start), 
                        np.abs(FT_intensity[idx_start: idx_end]), 
                        alpha = 0.2, 
                        color = "orange",
                        zorder = 0)
    
    plt.xlabel("Quasi-time (unitless)")
    if test_phase != None:
        plt.legend(["Reconstructed phase", "Initial phase", "FT intensity"], bbox_to_anchor = [0.95, -0.15])
    else:
        plt.legend(["Reconstructed phase", "FT intensity"], bbox_to_anchor = [0.95, -0.15])
    plt.grid()
    plt.savefig("pics/reconstructed{}.jpg".format(iter), bbox_inches = "tight", dpi = 200)
    plt.close()