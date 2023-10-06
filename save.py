
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
import spectral_analysis as sa

X = np.linspace(1,1824, num=1824)

for i in range(512):
    Y = sa.gaussian_pulse((1540,1560), 1550, 1, x_type='freq').Y# + np.random.uniform(low=0, high=0.01, size=1824)
    Y *=  1/np.sum(np.abs(Y))
    data = pd.DataFrame(np.transpose(np.vstack([Y])))
    data.to_csv(f'data_gauss/signal_{i}', index = False, header=None)

plt.plot(Y)
plt.savefig('gauss.png')
plt.close()
    
for i in range(512):
    Y = sa.hermitian_pulse(1, (1540,1560), 1550, 1, x_type='freq').Y#+ np.random.uniform(low=0, high=0.01, size=1824)
    Y *=  1/np.sum(np.abs(Y))
    Y = np.abs(Y)
    data = pd.DataFrame(np.transpose(np.vstack([Y])))
    data.to_csv(f'data_hermit/signal_{i}', index = False, header=None)

plt.plot(Y)
plt.savefig('hermit.png')
plt.close()
