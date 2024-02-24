import numpy as np
import torch
import torch.cuda as cuda
import matplotlib.pyplot as plt
from pyunlocbox import functions, solvers

def tv_regularization_unwrap_1d_gpu(phase):
    # Move data to GPU
    phase_gpu = torch.tensor(phase, dtype=torch.float32).cpu()

    # Set the TV regularization strength (adjust as needed)
    alpha = 0.2

    # Define the TV regularization term
    phi = functions.dummy()
    phi._prox = lambda x, gamma: functions.prox_tv(x, alpha * gamma)

    # Create the data fidelity term
    f = functions.norm_l2()



    # Solve the optimization problem using the Douglas-Rachford algorithm
    x0 = torch.zeros_like(phase_gpu)
    solver = solvers.douglas_rachford(step=1e-2)
    x = solvers.solve([f, phi], x0)

    # Move data back to CPU
    unwrapped_phase_cpu = x.cpu().numpy()

    return unwrapped_phase_cpu

# Example usage:
wrapped_phase = np.angle(np.exp(1j * np.linspace(0, 6 * np.pi, 1000)))  # Wrapped phase
unwrapped_phase = tv_regularization_unwrap_1d_gpu(wrapped_phase)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(wrapped_phase, label='Wrapped Phase')
plt.plot(unwrapped_phase, label='Unwrapped Phase')
plt.title('Wrapped and Unwrapped Phase (TV Regularization) on GPU')
plt.xlabel('Sample Index')
plt.ylabel('Phase')
plt.legend()
plt.show()
