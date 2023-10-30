def evolve(intensity, phase, device, dtype, abs = True,):

    import torch
    from math import floor

    input_dim = intensity.numel()
    output_dim = phase.numel()

    intensity = torch.fft.fftshift(intensity)
    intensity = torch.fft.fft(intensity)
    intensity = torch.fft.fftshift(intensity)
    
    long_phase = torch.concat([torch.zeros(size = [floor((input_dim-output_dim)/2)], requires_grad = True, device = device, dtype = dtype), 
                          phase,
                          torch.zeros(size = [floor((input_dim-output_dim)/2)], requires_grad = True, device = device, dtype = dtype)])
    
    complex_intensity = torch.mul(intensity, torch.exp(1j*long_phase))

    complex_intensity = torch.fft.ifftshift(complex_intensity)
    complex_intensity = torch.fft.ifft(complex_intensity)
    complex_intensity = torch.fft.ifftshift(complex_intensity)

    if abs:
        return complex_intensity.abs()
    else:
        return complex_intensity
    
    
def plot_phases(phase_generator, num, phase_type = "regular"):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    if not os.path.isdir("example_phases"):
        os.mkdir("example_phases")

    for i in range(num):
        phase = phase_generator(100, phase_type = phase_type)
        plt.plot(np.linspace(0, 1, 100), phase, color = "deeppink")
        plt.grid()
        plt.title("Test phase")
        plt.savefig("example_phases/phase_{}.jpg".format(i + 1))
        plt.close()


def np_to_complex_pt(array, device, dtype):
    '''
    # real-valued numpy array to complex pytorch tensor
    '''
    import torch

    array = torch.tensor([[array[i], 0] for i in range(len(array))], requires_grad = True, device = device, dtype = dtype)
    array = torch.view_as_complex(array)
    array = array.reshape(1, array.numel())

    return array