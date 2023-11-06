import torch
import numpy as np
import spectral_analysis as sa
from utilities import np_to_complex_pt, evolve_pt, evolve_np
import os
from tqdm import tqdm
import shutil

class Generator():

    def __init__(self, data_num, initial_intensity, phase_len, device, dtype, max_order=10, max_value=10):
        self.data_num = data_num
        self.initial_intensity = initial_intensity
        self.intensity_len = len(initial_intensity)
        self.phase_len = phase_len
        self.max_order = max_order
        self.max_value = max_value
        self.device = device
        self.dtype = dtype


    def generate_and_save(self):
        shutil.rmtree('data')

        if not os.path.isdir("data"):
            os.mkdir("data")
        if not os.path.isdir("data/train_intensity"):
            os.mkdir("data/train_intensity")
        if not os.path.isdir("data/train_phase"):
            os.mkdir("data/train_phase")

        for example_num in tqdm(range(self.data_num)):
            intensity, phase = self.pulse_gen()
            np.savetxt("data/train_intensity/" + str(example_num) + ".csv", intensity)
            np.savetxt("data/train_phase/" + str(example_num) + ".csv", phase)


    def phase_gen(self):
        if np.random.choice(5) == 1:      # slowly varying phase
            X = np.linspace(-1, 1, self.phase_len)
            Y = np.zeros(self.phase_len)
            
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
            Y = np.zeros(self.phase_len)
            for order in range(4):
                coef = np.random.uniform(low = -1, high = 1)
                Y += coef*sa.hermitian_pulse(pol_num = order,
                    bandwidth = [-1, 1],
                    centre = 0,
                    FWHM = 0.5,
                    num = self.phase_len).Y
        if self.max_value == None:
            return Y
        else:
            return Y/np.max(np.abs(Y))*self.max_value
        
    def pulse_gen(self):
        '''
        Returns tuple (intensity, phase), where phase is NONZERO part of phase used to evolve initial_intensity into intensity.
        '''

        intensity = self.initial_intensity.copy()
        intensity = np.array([complex(intensity[i], 0) for i in range(len(intensity))])
        
        phase_significant = self.phase_gen()
        
        intensity = evolve_np(intensity, phase_significant, dtype = self.dtype)
        
        return np.abs(intensity), phase_significant