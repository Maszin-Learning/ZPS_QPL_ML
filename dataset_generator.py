import torch
import numpy as np
import spectral_analysis as sa
from utilities import np_to_complex_pt, evolve_pt, evolve_np
import os
from tqdm import tqdm
import shutil
from math import floor

class Generator():

    def __init__(self, data_num, initial_intensity, phase_len, device, dtype, max_order=10, max_value=np.pi):
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

        def polynomial():
            '''
            Sum of regular polynomials.
            '''
            X = np.linspace(-1, 1, self.phase_len)
            Y = np.zeros(self.phase_len)

            max_order = 10
            
            for order in range(max_order):
                coef = np.random.uniform(low = -1, high = 1)
                Y += coef*X**order

            Y /= np.max(np.abs(Y))
            return Y
            
        def absolute_like():
            '''
            Absolute value with random "middle".
            '''
            X = np.linspace(-1, 1, self.phase_len)
            middle = np.random.uniform(-0.5, 0.5)
            X += middle
            return np.abs(X)/np.max(np.abs(X))

        def absolute_like_multi():
            '''
            Sum of up to 5 absolute values.
            '''
            Y = absolute_like()
            num = np.random.randint(5)
            for i in range(num):
                Y += absolute_like()
            Y /= np.max(np.abs(Y))
            return Y

        def step_like():
            '''
            Random constant value till some place, then another random constant value.
            '''
            a, b = np.random.uniform(low = 0, high = 2*np.pi, size = 2)
            border = np.random.randint(low = floor(0.25*self.phase_len), high = floor(0.75*self.phase_len))
            Y = np.concatenate([a*np.ones(border), b*np.ones(self.phase_len-border)])
            Y /= np.max(np.abs(Y))
            return Y
        
        def step_like_multi():
            '''
            Sum of up to 5 step functions.
            '''
            Y = step_like()
            num = np.random.randint(5)
            for i in range(num):
                Y += step_like()
            Y /= np.max(np.abs(Y))
            return Y

        def hermite_like():
            '''
            Sum of the Hermite polynomials.
            '''

            max_order = 6

            Y = np.zeros(self.phase_len)
            for order in range(max_order):
                coef = np.random.uniform(low = -1, high = 1)
                Y += coef*sa.hermitian_pulse(pol_num = order,
                    bandwidth = [-1, 1],
                    centre = 0,
                    FWHM = 0.5,
                    num = self.phase_len).Y
                
            Y /= np.max(np.abs(Y))
            return Y
        
        # let's toss a coin, my friend
        coin = np.random.choice(np.array([0,1,2,3,4,5]), size = 1, p = [0.35, 0.25, 0.10, 0.05, 0.20, 0.05])

        if coin == 0:
            phase = hermite_like()
        elif coin == 1:
            phase = polynomial()
        elif coin == 2:
            phase = step_like()
        elif coin == 3:
            phase = step_like_multi()
        elif coin == 4:
            phase = absolute_like()*5
        elif coin == 5:
            phase = absolute_like_multi()*5
        else:
            raise Exception("Your multidimensional coin has more dimensions that one could expect.")

        scale = np.random.uniform(1, np.pi)

        return phase*scale


    def pulse_gen(self):
        '''
        Returns tuple (intensity, phase), where phase is NONZERO part of phase used to evolve initial_intensity into intensity.
        '''

        intensity = self.initial_intensity.copy()
        intensity = np.array([complex(intensity[i], 0) for i in range(len(intensity))])
        
        phase_significant = self.phase_gen()
        
        intensity = evolve_np(intensity, phase_significant, dtype = self.dtype)
        
        return np.abs(intensity), phase_significant