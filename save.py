import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(1,1824, num=1824)
def gauss(x, mu, std):
    return 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*std**2))
for i in range(2048):
    Y = gauss(x, 900, 400) * 10000 + np.random.uniform(low=0, high=0.2, size=1824)

    data = pd.DataFrame(np.transpose(np.vstack([Y])))
    data.to_csv(f'data/signal_{i}', index = False)

plt.plot(Y)
plt.savefig('gauss.png')
    