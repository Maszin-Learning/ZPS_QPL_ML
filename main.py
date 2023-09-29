import spectral_analysis as sa
import numpy as np

pulse_1 = sa.gaussian_pulse((1540,1560), 1550, 3, x_type='freq')
pulse_1.x_type = "wl"
pulse_1.wl_to_freq()
pulse_1.Y = pulse_1.Y*100
signal_len=len(pulse_1)
sa.plot(pulse_1, title = 'przed_1', save = True)

pulse_2 = sa.hermitian_pulse((1545,1565), 1555, 3, x_type='freq')
pulse_2.x_type = "wl"
pulse_2.wl_to_freq()
pulse_2.Y = pulse_2.Y*100

#sa.plot(pulse_1)

#pulse_1.fourier()
#sa.plot(pulse_1)
#pulse_1.Y = pulse_1.Y*np.exp(1j*pulse_1.X)
#pulse_1.inv_fourier()   
#sa.plot(pulse_1, title='' , save=True)



### OPTIMALIZATION
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device_ = torch.device("cuda")
    print (f"Using {device_}")
    #Checking GPU RAM allocated memory
    print('allocated CUDA memory: ',torch.cuda.memory_allocated())
    print('cached CUDA memory: ',torch.cuda.memory_cached())
    torch.cuda.empty_cache() # clear CUDA memory
    torch.backends.cudnn.benchmark = True #let cudnn chose the most efficient way of calculating Convolutions
    
elif torch.backends.mps.is_available():
    print ("CUDA device not found.")
    device_ = torch.device("mps")
    print (f"Using {device_}")
else:
    print ("MPS device not found.")
    device_ = torch.device("cpu")
    print (f"Using {device_}")

#define globaly used dtype and device
device_ = torch.device('cpu')
dtype_ = torch.float32

# create class
class AutoEncoder(nn.Module):
    def __init__(self,input_size,n,output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super (AutoEncoder,self).__init__()
        # Linear function.
        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(n,output_size)
        self.leakyrelu=nn.LeakyReLU(1, inplace=True)
        self.normal_1 = nn.LayerNorm()

    def forward(self,x):
        x = self.leakyrelu(self.linear_1(x))
        #x = self.leakyrelu(self.linear_2(x))
        return self.leakyrelu(self.linear_3(x))
    
#configuration of W&B
config = dict (
    # Hyper-parameters
    input_dim = 32, # and dim of noise vector
    output_dim = signal_len,
    p = 5, #number of plots
    criterion = nn.L1Loss(), # loss function jest beznajdziejna
    learning_rate = 1e-9,
    epoch_num = 40000,
    node_number = 50,
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
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.01, betas=(0.9, 0.999))
# Optimization (find parameters that minimize error)
# X = random phase input, shape = (2*signal input, 1)
# Y = defined shape of function

#X_real=pulse_1.X.real
#X_imag=pulse_1.X.imag
#X_pulse_conc= np.concatenate((X_real, X_imag), axis=None)
#X = torch.tensor(X_pulse_conc.reshape(signal_len*2,1), requires_grad=True, device=device_).to(dtype=dtype_)
X = torch.tensor(np.random.uniform(low=0,high=1,size=(1, config['input_dim'])), requires_grad=True, device=device_, dtype=dtype_)


Y_real=pulse_2.Y.real
Y_imag=pulse_2.Y.imag
Y_pulse_conc= np.concatenate((Y_real, Y_imag), axis=None)
Y = torch.tensor(Y_pulse_conc.reshape(signal_len*2,1), requires_grad=True, device=device_, dtype=dtype_)



    
#calculation
loss_list = []
z_ = config['epoch_num']/config['p']

parameters ={}
criterion = config['criterion']

pulse_1.fourier()

pulse_2_Y_real=pulse_2.Y.real
pulse_2_Y_imag=pulse_2.X.imag
pulse_2_Y_abs_tensor = torch.tensor(np.abs(pulse_2.Y), requires_grad=True, device=device_, dtype=dtype_).reshape(1,signal_len)
pulse_2_conc_target = np.concatenate((pulse_2_Y_real, pulse_2_Y_imag), axis=None)
pulse_2_conc_target_torch = torch.tensor(pulse_2_conc_target.reshape(1,signal_len*2), requires_grad=True, device=device_, dtype=dtype_)



for epoch in tqdm(range(config['epoch_num'])):
    
    optimizer.zero_grad() # zero gradients
    results = model(X) # Forward to get output
    #print(results.shape)
    x = torch.tensor([pulse_1.Y.real, pulse_1.Y.imag], requires_grad=True, device=device_, dtype=dtype_)
    x = x.reshape(signal_len, 2)
    pulse_1_torch_Y = torch.view_as_complex(x)
    pulse_1_torch_Y = pulse_1_torch_Y.reshape(1, signal_len)
    #print(pulse_1_torch_Y.shape)
    #print(torch.exp(1j*results).shape)
    pulse_1_torch_Y = torch.mul(pulse_1_torch_Y, torch.exp(1j*results))
    pulse_1_torch_Y = torch.fft.ifft(pulse_1_torch_Y)
    pusle_1_Y_abs_tensor = pulse_1_torch_Y.abs()
    #pulse_1_conc_result_torch= torch.concatenate((pulse_1_torch_Y.real, pulse_1_torch_Y.imag), axis=1)
    loss = criterion(pusle_1_Y_abs_tensor, pulse_2_Y_abs_tensor) # Calculate Loss/criterion
    
    loss.backward() # backward propagation
    optimizer.step() # Updating parameters
    loss_list.append(loss.data) # store loss
    
    # print loss
    if epoch % 500 == 0:
        pulse_1.Y = pulse_1.Y*np.exp(1j*results.clone().detach().numpy().reshape(signal_len,))
        pulse_1.inv_fourier()   
        sa.plot(pulse_1, title=f'reconstructed_{epoch}' , save=True)
        pulse_1.fourier()
        print('epoch {}, loss {}'.format(epoch, loss.data))

#print(f'---Model calculated---\nloss: {loss_list[config['epoch_num'] - 1]}')





pulse_1.Y = pulse_1.Y*np.exp(1j*results.detach().numpy().reshape(signal_len,))
pulse_1.inv_fourier()   
sa.plot(pulse_1, title='reconstructed' , save=True)
sa.plot(pulse_2, title='target' , save=True)
