import torch
import numpy as np
import spectral_analysis as sa
from utilities import np_to_complex_pt, evolve

class Generator():

    def __init__(self, batch_num, batch_size, initial_intensity, phase_len, device, dtype, max_order=10, max_value=10):
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.initial_intensity = initial_intensity
        self.num = len(initial_intensity)
        self.phase_num = phase_len
        self.max_order = max_order
        self.max_value = max_value
        self.device = device
        self.dtype = dtype  
        self.data = [self.pulse_gen() for i in range(batch_size*batch_num)]
        
    def phase_gen(self):
        if np.random.choice(5) == 1:      # slowly varying phase
            X = np.linspace(-1, 1, self.phase_num)
            Y = np.zeros(self.phase_num)
            
            for order in range(self.max_order):
                coef = np.random.uniform(low = -1, high = 1)

                if np.random.choice(50) == 1:
                    _coef = np.random.uniform(low = -1, high = 1)
                    Y += coef*X**order + _coef * np.sin(X)
                if np.random.choice(50) == 1:
                    _coef = np.random.uniform(low = -1, high = 1)
                    Y += coef*X**order + _coef * np.tan(X)
                if np.random.choice(50) == 1:
                    Y += coef*X**order * np.tan(X)
                if np.random.choice(50) == 1:
                    Y += coef*X**order * np.sin(X)
                else:
                    Y += coef*X**order
                    
                """
                if r == 1:
                    Y += np.tan(X)*coef*X**order
                if r == 2:
                    Y += np.sin(X)*coef*X**order
                """
        else:                                               # rapidly varying phase  UPDATE: It causes convergence
            Y = np.zeros(self.phase_num)
            for order in range(4):
                coef = np.random.uniform(low = -1, high = 1)
                Y += coef*sa.hermitian_pulse(pol_num = order,
                    bandwidth = [-1, 1],
                    centre = 0,
                    FWHM = 0.5,
                    num = self.phase_num).Y
        if self.max_value == None:
            return Y
        else:
            return Y/np.max(np.abs(Y))*self.max_value
        
    def pulse_gen(self):

        intensity = self.initial_intensity.copy()
        intensity = np_to_complex_pt(intensity, device = self.device, dtype = self.dtype)
        
        phase_significant = self.phase_gen()

        phase_significant = torch.tensor(phase_significant, requires_grad = True, device = self.device, dtype = self.dtype)
        
        intensity = evolve(intensity, phase_significant, device = self.device, dtype = self.dtype)
        
        return intensity.abs(), phase_significant