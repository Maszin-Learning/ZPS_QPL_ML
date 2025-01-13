import torch.nn as nn
import torch
import numpy as np
import torchaudio

#duble transformations

class network_2_1(nn.Module): #spectral network
    def __init__(self, input_size, n, output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(network_2_1, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(n,output_size)
        self.tanh = nn.Tanh()
        self.leakyrelu=nn.LeakyReLU(1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(input_size)
        self.normal_3 = nn.LayerNorm(output_size)

    def forward(self,x):
        x = self.normal_1(x)
        x = self.leakyrelu(self.linear_1(x))
        x = self.linear_3(x)

        return self.tanh(x)* np.pi*6
    
class network_2_2(nn.Module): #temporal network
    def __init__(self, input_size, n, output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(network_2_2, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(n,output_size)
        self.tanh = nn.Tanh()
        self.leakyrelu=nn.LeakyReLU(1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(input_size)
        self.normal_3 = nn.LayerNorm(output_size)

    def forward(self,x):
        x = self.normal_1(x)
        x = self.leakyrelu(self.linear_1(x))
        x = self.linear_3(x)

        return self.tanh(x)* np.pi*10
    
    
### convs   
    
class network_3_1(nn.Module):
    def __init__(self, input_size, n, output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(network_3_1, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(240,output_size)
        self.tanh = nn.Tanh()
        self.leakyrelu=nn.LeakyReLU(1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(input_size)
        self.normal_3 = nn.LayerNorm(output_size)
        
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                            out_channels=20,
                            kernel_size=20,
                            stride=4,
                            padding=5)
        
        self.conv1d_2 = nn.Conv1d(in_channels=20,
                    out_channels=20,
                    kernel_size=5,
                    stride=2,
                    padding=2)
        
        self.linear_01 = nn.Linear(n,output_size)

    def forward(self,x):

        x = torch.unsqueeze(x, 1)
        x = x.reshape(1,-1)
        x = self.normal_1(x)
        x = self.leakyrelu(self.linear_1(x))
        x_0 = self.leakyrelu(self.linear_01(x))
        x = self.sigmoid(self.conv1d_1(x))
        x = self.sigmoid(self.conv1d_2(x))
        print(x.shape)
        x = torch.flatten(x)
        print(x.shape)
        x = self.linear_3(x)
        #x = self.normal_3(x)
        x = torch.squeeze(x+ 1*torch.squeeze(x_0))
        return self.tanh(x)* np.pi*6
    
class network_3_2(nn.Module):
    def __init__(self, input_size, n, output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(network_3_2, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(240,output_size)
        self.tanh = nn.Tanh()
        self.leakyrelu=nn.LeakyReLU(1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(input_size)
        self.normal_3 = nn.LayerNorm(output_size)
        
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                            out_channels=20,
                            kernel_size=20,
                            stride=4,
                            padding=5)
        
        self.conv1d_2 = nn.Conv1d(in_channels=20,
                    out_channels=20,
                    kernel_size=5,
                    stride=2,
                    padding=2)
        
        self.linear_01 = nn.Linear(n,output_size)

    def forward(self,x):

        x = torch.unsqueeze(x, 1)
        x = x.reshape(1,-1)
        x = self.normal_1(x)
        x = self.leakyrelu(self.linear_1(x))
        x_0 = self.leakyrelu(self.linear_01(x))
        x = self.sigmoid(self.conv1d_1(x))
        x = self.sigmoid(self.conv1d_2(x))
        print(x.shape)
        x = torch.flatten(x)
        print(x.shape)
        x = self.linear_3(x)
        #x = self.normal_3(x)
        x = torch.squeeze(x+ 1*torch.squeeze(x_0))
        return self.tanh(x)* np.pi*10
    
### convs  with FFT 
    
class network_4_1(nn.Module):
    def __init__(self, input_size, n, output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(network_3_1, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(240,output_size)
        self.tanh = nn.Tanh()
        self.leakyrelu=nn.LeakyReLU(1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(input_size)
        self.normal_3 = nn.LayerNorm(output_size)
        
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                            out_channels=20,
                            kernel_size=20,
                            stride=4,
                            padding=5)
        
        self.conv1d_2 = nn.Conv1d(in_channels=20,
                    out_channels=20,
                    kernel_size=5,
                    stride=2,
                    padding=2)
        
        self.linear_01 = nn.Linear(n,output_size)

    def forward(self, x, FFT):
        #FFT transformation
        x = torch.unsqueeze(x, 1)
        x = x.reshape(1,-1)
        
        #main signal transformation
        x = torch.unsqueeze(x, 1)
        x = x.reshape(1,-1)
        x = self.normal_1(x)
        x = self.leakyrelu(self.linear_1(x))
        x_0 = self.leakyrelu(self.linear_01(x))
        x = self.sigmoid(self.conv1d_1(x))
        x = self.sigmoid(self.conv1d_2(x))
        x = torch.flatten(x)
        x = self.linear_3(x)
        #x = self.normal_3(x)
        x = torch.squeeze(x+ 1*torch.squeeze(x_0))
        return self.tanh(x)* np.pi*6
    
class network_4_2(nn.Module):
    def __init__(self, input_size, n, output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(network_3_2, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(240,output_size)
        self.tanh = nn.Tanh()
        self.leakyrelu=nn.LeakyReLU(1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(input_size)
        self.normal_3 = nn.LayerNorm(output_size)
        
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                            out_channels=20,
                            kernel_size=20,
                            stride=4,
                            padding=5)
        
        self.conv1d_2 = nn.Conv1d(in_channels=20,
                    out_channels=20,
                    kernel_size=5,
                    stride=2,
                    padding=2)
        
        self.linear_01 = nn.Linear(n,output_size)

    def forward(self, x, FFT):

        x = torch.unsqueeze(x, 1)
        x = x.reshape(1,-1)
        x = self.normal_1(x)
        x = self.leakyrelu(self.linear_1(x))
        x_0 = self.leakyrelu(self.linear_01(x))
        x = self.sigmoid(self.conv1d_1(x))
        x = self.sigmoid(self.conv1d_2(x))
        x = torch.flatten(x)
        x = self.linear_3(x)
        #x = self.normal_3(x)
        x = torch.squeeze(x+ 1*torch.squeeze(x_0))
        return self.tanh(x)* np.pi*10