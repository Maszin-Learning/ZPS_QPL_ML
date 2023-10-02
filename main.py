# modules

import spectral_analysis as sa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# define initial and target pulse

initial_pulse = sa.gaussian_pulse((190, 196), 193, 1, x_type ='freq')
target_pulse = sa.hermitian_pulse((190, 196), 193, 1, x_type ='freq')

target_pulse.Y *= np.sqrt((np.sum((initial_pulse.Y)**2))/np.sum((target_pulse.Y)**2))

signal_len = len(initial_pulse)

plt.plot(initial_pulse.X, initial_pulse.Y, color = "red")
plt.grid()
plt.title("Initial pulse")
plt.xlabel("Frequency (THz)")
plt.show()

plt.plot(target_pulse.X, target_pulse.Y, color = "blue")
plt.grid()
plt.title("Target pulse")
plt.xlabel("Frequency (THz)")
plt.show()

### CUDA & stuff

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device_ = torch.device("cuda")
    print (f"Using {device_}")
    print('allocated CUDA memory: ',torch.cuda.memory_allocated())      # Checking GPU RAM allocated memory
    print('cached CUDA memory: ',torch.cuda.memory_cached())
    torch.cuda.empty_cache()                                            # clear CUDA memory
    torch.backends.cudnn.benchmark = True                               # let cuda chose the most efficient way of calculating Convolutions
    
elif torch.backends.mps.is_available():
    print ("CUDA device not found.")
    device_ = torch.device("mps")
    print (f"Using {device_}")
else:
    print ("MPS device not found.")
    device_ = torch.device("cpu")
    print (f"Using {device_}")

# define globaly used datatype and device

device_ = torch.device('cpu')
dtype_ = torch.float32

# define neural network

class AutoEncoder(nn.Module):
    def __init__(self,input_size,n,output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super (AutoEncoder,self).__init__()

        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(n,output_size)
        
        self.leakyrelu=nn.LeakyReLU(1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)

    def forward(self,x):
        x = self.leakyrelu(self.linear_1(x))
        x = self.normal_1(x)
        #x = self.leakyrelu(self.linear_2(x))
        x = self.linear_3(x)
        x = self.normal_3(x)
        return self.leakyrelu(x)
    
# HyperParameters

config = dict(
    input_dim = 32,             # and dim of noise vector
    output_dim = 50,            # signal_len,
    p = 5,                      # number of plots
    criterion = nn.MSELoss(),  
    learning_rate = 5e-7,
    epoch_num = 100000,
    node_number = 100,
    architecture = "NN_1",
    dataset_id = "peds-0192",
    infra = "Local_cpu",
)    
    
# Initialize network

model = AutoEncoder(input_size=config['input_dim'], n=config['node_number'], output_size=config['output_dim'])
total_params = sum(k.numel() for k in model.parameters())
print(f"Number of parameters: {total_params}")
print(f'Number of nodes: {config["node_number"]}')
model.to(device=device_, dtype=dtype_) # project model onto device and chosen dtype
optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'], weight_decay = 0.01, betas = (0.9, 0.999))

# random phase to start with something

noise_phase = torch.tensor(np.random.uniform(low = 0, high = 1,
                                    size = (1, config['input_dim'])), 
                                    requires_grad=True, 
                                    device = device_, 
                                    dtype = dtype_)


# is it necessary?

loss_list = []
z_ = config['epoch_num']/config['p']

parameters = {}
criterion = config['criterion']

# initial and target pulses

initial_X = initial_pulse.X.copy()
initial_Y = initial_pulse.Y.copy()
initial_real = initial_Y.real
initial_imag = initial_Y.imag

target_real = target_pulse.Y.real
target_imag = target_pulse.X.imag
target_abs = torch.tensor(np.abs(target_pulse.Y), requires_grad=True, device=device_, dtype=dtype_).reshape(1, signal_len)

# learning loop

for epoch in tqdm(range(config['epoch_num'])):
    
    # net predictions

    optimizer.zero_grad()           # zero gradients
    results = model(noise_phase)    # Forward to get output
    results = torch.reshape(results, [50, 1])
    phase = torch.concat([torch.zeros([975, 1], requires_grad = True), 
                          results, 
                          torch.zeros([975, 1], requires_grad = True)])
    # get intensity spectrum

    initial_imag_and_real = torch.tensor([[initial_Y.real[i], initial_Y.imag[i]] for i in range(len(initial_pulse))], 
                                         requires_grad = True, device = device_, dtype = dtype_)
    complex_initial = torch.view_as_complex(initial_imag_and_real)
    complex_initial = complex_initial.reshape(1, signal_len)
    complex_initial = torch.fft.fft(complex_initial)

    # create complex spectrum

    phase = torch.fft.fftshift(phase)
    phase = phase.reshape([1, 2000])
    complex_initial = torch.mul(complex_initial, torch.exp(1j*phase))

    # IFT

    complex_initial = torch.fft.ifft(complex_initial)

    # loss & optimizer step

    reconstructed = complex_initial.abs()
    loss = criterion(reconstructed, target_abs)
    loss.backward() # backward propagation
    optimizer.step() # Updating parameters
    #loss_list.append(loss.data) # store loss
    
    # print loss

    if epoch % 500 == 0:
        if epoch % 500 == 0:
            plt.close()
            plt.plot(initial_pulse.X, np.reshape(reconstructed.clone().detach().numpy(), 2000), color = "green")
            plt.plot(initial_pulse.X, np.reshape(target_abs.clone().detach().numpy(), 2000), color = "orange")
            plt.grid()
            plt.savefig("pics/reconstructed{}.jpg".format(epoch))
            plt.close()
        
        print('epoch {}, loss {}'.format(epoch, loss.data))