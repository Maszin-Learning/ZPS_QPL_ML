import torch
import numpy as np
import spectral_analysis as sa
from utilities import np_to_complex_pt, evolve_pt, evolve_np, shift_to_centre
import os
import test
from tqdm import tqdm
import shutil
from math import floor

class Generator():

    def __init__(self, data_num, initial_intensity, FT_X, phase_len, device, dtype, max_order=10, max_value=np.pi):
        self.data_num = data_num
        self.initial_intensity = initial_intensity
        self.FT_X = FT_X
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

        for example_num in tqdm(range(1, self.data_num + 1)):
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

            max_order = 5
            
            for order in range(max_order):
                coef = np.random.uniform(low = -1, high = 1)
                Y += coef*X**order

            Y /= np.max(np.abs(Y))
            return Y*np.random.uniform(1, 2*np.pi)
            
        def absolute_like():
            '''
            Absolute value with random "middle".
            '''
            X = np.linspace(-1, 1, self.phase_len)
            middle = 0#np.random.uniform(-0.5, 0.5)
            X += middle
            Y = np.abs(X)/np.max(np.abs(X))
            return Y*np.random.uniform(1, 25*np.pi)

        def absolute_like_multi():
            '''
            Sum of up to 5 absolute values.
            '''
            Y = absolute_like()
            num = np.random.randint(5)
            for i in range(num):
                Y += absolute_like()
            Y /= np.max(np.abs(Y))
            return Y*np.random.uniform(1, 25*np.pi)

        def step_like():
            '''
            Random constant value till some place, then another random constant value.
            '''
            a, b = np.random.uniform(low = 0, high = 2*np.pi, size = 2)
            border = np.random.randint(low = floor(0.3*self.phase_len), high = floor(0.7*self.phase_len))
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

            max_order = 15

            Y = np.zeros(self.phase_len)
            for order in range(max_order):
                coef = np.random.uniform(low = -1, high = 1)
                Y += coef*sa.hermitian_pulse(pol_num = order,
                    bandwidth = [-1, 1],
                    centre = 0,
                    FWHM = 0.5,
                    num = self.phase_len,
                    broad = True).Y
                
            Y /= np.max(np.abs(Y))
            return Y*np.random.uniform(1, 5*np.pi)
        
        # let's toss a coin, my friend
        coin = np.random.choice(np.array([0,1,2,3,4,5]), size = 1, p = [1, 0.00, 0.00, 0.00, 0.00, 0.00])

        if coin == 0:
            phase = hermite_like()
        elif coin == 1:
            phase = polynomial()
        elif coin == 2:
            phase = step_like()
        elif coin == 3:
            phase = step_like_multi()
        elif coin == 4:
            phase = absolute_like()
        elif coin == 5:
            phase = absolute_like_multi()
        else:
            raise Exception("Your multidimensional coin has more dimensions that one could expect.")
        
        '''
        shift = np.random.uniform(-1, 1) # just random shift from -1 up to 1 THz
        phase += 2*np.pi*shift*self.FT_X
        '''

        return phase


    def pulse_gen(self):
        '''
        Returns tuple (intensity, phase), where phase is NONZERO part of phase used to evolve initial_intensity into intensity.
        '''

        intensity = self.initial_intensity.copy()
        intensity = np.array([complex(intensity[i], 0) for i in range(len(intensity))])
        probability = np.random.uniform(0, 1)

        if probability < 0.5: # phases from the phase generator
            phase_significant = self.phase_gen()
            intensity = evolve_np(intensity, phase_significant, dtype = self.dtype)

        elif True:    # exponential
            parameter = np.random.uniform(1, 20)
            intensity = np.exp(np.linspace(-3, parameter, self.intensity_len)) - np.exp(-1.5)
            for i in range(floor(len(intensity)*3/4), len(intensity)):
                intensity[i] = 0
            for i in range(0, floor(len(intensity)*1/4)):
                intensity[i] = 0
            intensity = intensity / np.sqrt(np.sum(intensity*np.conjugate(intensity)))
            intensity = intensity * np.sqrt(np.sum(self.initial_intensity*np.conjugate(self.initial_intensity)))
            phase_significant = np.ones(self.phase_len)

        elif False: # hermite intensities
            order = np.random.randint(5)
            correction = np.random.uniform(-0.5, 0.5)
            intensity = sa.hermitian_pulse(pol_num = order,
                                           bandwidth = [190, 196],
                                           centre = 193,
                                           FWHM = 1,
                                           num = len(intensity)).Y
            
            intensity = intensity / np.sqrt(np.sum(intensity*np.conjugate(intensity)))
            intensity = intensity * np.sqrt(np.sum(self.initial_intensity*np.conjugate(self.initial_intensity)))
            phase_significant = np.ones(self.phase_len)

        elif False: # gauss intensities
            
            correction_1 = np.random.uniform(-0.5, 0.5)
            correction_2 = np.random.uniform(-0.5, 0.5)

            intensity = sa.hermitian_pulse(pol_num = 0,
                                bandwidth = [190, 196],
                                centre = 193,
                                FWHM = 1 + correction_2,
                                num = len(intensity)).Y
            
            intensity = intensity / np.sqrt(np.sum(intensity*np.conjugate(intensity)))
            intensity = intensity * np.sqrt(np.sum(self.initial_intensity*np.conjugate(self.initial_intensity)))
            phase_significant = np.ones(self.phase_len)

        intensity = shift_to_centre(intensity_to_shift = intensity,
                                    intensity_ref = self.initial_intensity)
        
        return np.abs(intensity), phase_significant ### phase_significant is now wrong up to the linear phase