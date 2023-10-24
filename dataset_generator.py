import torch
import numpy as np
import spectral_analysis as sa

class Generator():
    def __init__(self, num, max_order, max_value=None):
        self.num = num
        self.max_order = max_order
        self.max_value = max_value
        
        
        
    def phase_gen(self):
        if np.random.choice(1) == 1:      # slowly varying phase
            X = np.linspace(-1, 1, self.num)
            Y = np.zeros(self.num)
            
            for order in range(self.max_order):
                coef = np.random.uniform(low = -1, high = 1)
                Y += coef*X**order
        else:                                               # rapidly varying phase  UPDATE: It causes convergence
            Y = np.zeros(self.num)
            for order in range(4):
                coef = np.random.uniform(low = -1, high = 1)
                Y += coef*sa.hermitian_pulse(pol_num = order,
                    bandwidth = [-1, 1],
                    centre = 0,
                    FWHM = 0.5,
                    num = self.num).Y
        if self.max_value == None:
            return Y
        else:
            return Y/np.max(np.abs(Y))*self.max_value