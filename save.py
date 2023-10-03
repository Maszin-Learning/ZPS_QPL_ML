import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
import spectral_analysis as sa

X = np.linspace(1,1824, num=1824)
def gauss(x, mu, std):
    return 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*std**2))
for i in range(8192):
    Y = sa.gaussian_pulse((1550,1560), 1555, 4, x_type='freq').Y + np.random.uniform(low=0, high=0.01, size=1824)
    Y *=  np.sqrt(1/np.sum((Y)**2))
    data = pd.DataFrame(np.transpose(np.vstack([Y])))
    data.to_csv(f'data_gauss/signal_{i}', index = False, header=None)

plt.plot(Y)
plt.savefig('gauss.png')
plt.close()
    
for i in range(8192):
    Y = sa.hermitian_pulse((1550,1560), 1555, 4, x_type='freq').Y+ np.random.uniform(low=0, high=0.01, size=1824)
    Y *=  np.sqrt(1/np.sum((Y)**2))
    Y += 0.04 
    data = pd.DataFrame(np.transpose(np.vstack([Y])))
    data.to_csv(f'data_hermit/signal_{i}', index = False, header=None)

plt.plot(Y)
plt.savefig('hermit.png')
plt.close()