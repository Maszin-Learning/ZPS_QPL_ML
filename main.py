import spectral_analysis as sa
import numpy as np

pulse_1 = sa.gaussian_pulse((1540,1560), 1550, 3)
pulse_1.x_type = "wl"
pulse_1.wl_to_freq()


#sa.plot(pulse_1)
pulse_1.x_type = "freq"
#pulse_1.fourier()
#sa.plot(pulse_1)
pulse_1.Y = pulse_1.Y*np.exp(1j*pulse_1.X)
pulse_1.fourier()
sa.plot(pulse_1, title='d' , save=True)