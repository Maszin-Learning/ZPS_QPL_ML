import torch.nn as nn
import torch
import numpy as np
import torchaudio

'''
Simple nonlinear networks
'''


class network_0(nn.Module):
    def __init__(self, input_size, n, output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(network_0, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(360,output_size)
        self.sigmoid = nn.Sigmoid()
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
        x = self.normal_1(x)
        x = self.leakyrelu(self.linear_1(x))
        x_0 = self.leakyrelu(self.linear_01(x))
        x = self.sigmoid(self.conv1d_1(x))
        x = self.sigmoid(self.conv1d_2(x))
        
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.linear_3(x)
        #x = self.normal_3(x)
        x = torch.squeeze(x+ torch.squeeze(x_0))
        return self.sigmoid(x)* np.pi*2
    
    



### network_1
class network_1(nn.Module):
    def __init__(self, input_size, n, output_size):
        super(network_1, self).__init__()
        self.input = input_size
        self.output = output_size
        
        self.linear_1 = nn.Linear(input_size, n)
        self.linear_2 = nn.Linear(n, n)
        self.linear_3 = nn.Linear(n, output_size)
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace = True)
        self.sigmoid = nn.Sigmoid()
        
        self.normal_1 = nn.LayerNorm(n)
        self.bn_1 = nn.BatchNorm1d(n) #wont work on cpu
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        #print(x.shape)
        x = self.leakyrelu(self.linear_1(x))
        x = self.bn_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        return self.sigmoid(x)* np.pi*2

    

### network_2 testowa dla cpu
class network_2(nn.Module):
    def __init__(self, input_size, n, output_size):
        super(network_2, self).__init__()
        self.input = input_size
        self.output = output_size

        self.linear_1 = nn.Linear(input_size, n)
        self.linear_2 = nn.Linear(n, n)
        self.linear_3 = nn.Linear(n, output_size)
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace = True)
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        x = self.leakyrelu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_3(x)
        return self.sigmoid(x) * np.pi*2
    

class network_3(nn.Module): #very good very promising at -nn 300 -lr 1e-4 -bs 2 -en 100
    def __init__(self, input_size, n, output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(network_3, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(1776,output_size)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu=nn.LeakyReLU(1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(input_size)
        self.normal_3 = nn.LayerNorm(output_size)
        
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                            out_channels=37,
                            kernel_size=20,
                            stride=4,
                            padding=5)
        
        self.softmax = nn.Softmax()
        
        

    def forward(self,x):
        x = torch.unsqueeze(x, 1)
        x = self.normal_1(x)
        x = self.leakyrelu(self.linear_1(x))
        x = self.sigmoid(self.conv1d_1(x))
        
        #print(x.shape)
        
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        x = self.linear_3(x)
        #x = self.normal_3(x)
        x = torch.squeeze(x)
        return self.sigmoid(x)* np.pi*2
    

### network_4
class network_4(nn.Module): 
    def __init__(self, input_size, n, output_size):
        super(network_4, self).__init__()
        self.input = input_size
        self.output = output_size

        self.linear_1 = nn.Linear(input_size, n)
        self.linear_2 = nn.Linear(n, n)
        self.linear_3 = nn.Linear(n, output_size)
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace = True)
        self.sigmoid = nn.Sigmoid()
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
        self.bn_1 = nn.BatchNorm1d(n) #wont work on cpu
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        #print(x.shape)
        x = self.leakyrelu(self.linear_1(x))
        x = self.bn_1(x)
        x = self.leakyrelu(self.linear_2(x))
        x = self.bn_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        return self.sigmoid(x)* np.pi*2
    
class network_11(nn.Module):
    def __init__(self, input_size, n, output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(network_11, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(n,output_size)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu=nn.LeakyReLU(1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(input_size)
        self.normal_3 = nn.LayerNorm(output_size)

    def forward(self,x):
        x = self.normal_1(x)
        x = self.leakyrelu(self.linear_1(x))
        x = self.linear_3(x)

        return self.sigmoid(x)* np.pi*2
    
class network_12(nn.Module):
    def __init__(self, input_size, n, output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(network_12, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(input_size,n)
        #self.linear_2 = nn.Linear(n,n)
        self.linear_3 = nn.Linear(n,output_size)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu=nn.LeakyReLU(1, inplace=True)
        
        self.normal_1 = nn.LayerNorm(input_size)
        self.normal_3 = nn.LayerNorm(output_size)

    def forward(self,x):
        x = self.normal_1(x)
        x = self.leakyrelu(self.linear_1(x))
        x = self.linear_3(x)
        x = (torch.roll(x,-3) + torch.roll(x,-2) + torch.roll(x,-1) + torch.roll(x, 0) + torch.roll(x,1) + torch.roll(x,2) + torch.roll(x,3)) /7
        return self.sigmoid(x)* np.pi*2
    
    
    
'''
Convolution networks
'''  
    
class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self._in_channels=in_channels
        self._out_channels=out_channels
        self._kernel_size=kernel_size
        self._stride=stride
        self._padding=padding
        self.conv1d = nn.Conv1d(in_channels=self._in_channels,
                                  out_channels=self._out_channels,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding)
        
        self.avg_pool1d = nn.AvgPool1d(kernel_size=3,
                                         stride=None,
                                         padding=0)

    def forward(self, x):
        x = self.avg_pool1d(self.conv1d(x))
        return x


### network_7 with convolutionas
class network_7(nn.Module): #do not work on cpu
    def __init__(self, input_size, n, output_size):
        super(network_7, self).__init__()
        self.input = input_size
        self.output = output_size

        self.linear_1 = nn.Linear(3920, n) # change 76 to scalable wersion
        self.linear_2 = nn.Linear(n, n)
        self.linear_3 = nn.Linear(n, output_size)
        
        #convolutions
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                                  out_channels=5,
                                  kernel_size=11,
                                  stride=1,
                                  padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=5,
                                  out_channels=5,
                                  kernel_size=7,
                                  stride=1,
                                  padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=5,
                                  out_channels=10,
                                  kernel_size=5,
                                  stride=1,
                                  padding=1)
        self.conv1d_4 = nn.Conv1d(in_channels=10,
                                  out_channels=20,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        
        self.max_pool1d_1 = nn.MaxPool1d(kernel_size=5,
                                         stride=None,
                                         padding=0)
        
        
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace = True)
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.bn_fc_1 = nn.BatchNorm1d(n) #wont work on cpu
        self.bn_fc_1 = nn.BatchNorm1d(n)
        self.dropout = nn.Dropout(0.25)
        

    def forward(self,x):
        x = torch.unsqueeze(x, 1)
        #print(x.shape)
        x = self.conv1d_1(x)
        x = self.max_pool1d_1(x)
        x = self.conv1d_2(x)
        x = self.max_pool1d_1(x)
        x = self.conv1d_3(x)  
        x = self.conv1d_4(x)        
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        #print(x.shape)
        x = self.relu(self.linear_1(x))
        x = self.bn_fc_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        
        return self.sigmoid(x)* np.pi*2
    

    
    
### network_8 with convolutionas
class network_8(nn.Module): #do not work on cpu
    def __init__(self, input_size, n, output_size):
        super(network_8, self).__init__()
        self.input = input_size
        self.output = output_size

        self.linear_1 = nn.Linear(1520, n) # change 76 to scalable wersion
        self.linear_2 = nn.Linear(n, n)
        self.linear_3 = nn.Linear(n, output_size)
        
        #convolutions
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                                  out_channels=5,
                                  kernel_size=11,
                                  stride=1,
                                  padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=5,
                                  out_channels=10,
                                  kernel_size=7,
                                  stride=1,
                                  padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=10,
                                  out_channels=20,
                                  kernel_size=5,
                                  stride=1,
                                  padding=1)
        self.conv1d_4 = nn.Conv1d(in_channels=20,
                                  out_channels=20,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        
        self.max_pool1d_1 = nn.MaxPool1d(kernel_size=5,
                                         stride=None,
                                         padding=0)
        
        
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace = True)
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.bn_fc_1 = nn.BatchNorm1d(n) #wont work on cpu
        self.bn_fc_1 = nn.BatchNorm1d(n)
        self.dropout = nn.Dropout(0.25)
        

    def forward(self,x):
        x = torch.unsqueeze(x, 1)
        #print(x.shape)
        x = self.conv1d_1(x)
        x = self.max_pool1d_1(x)
        x = self.conv1d_2(x)
        x = self.max_pool1d_1(x)
        x = self.conv1d_3(x)  
        x = self.conv1d_4(x)        
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        #print(x.shape)
        x = self.relu(self.linear_1(x))
        x = self.bn_fc_1(x)
        x = self.dropout(x)
        x = self.relu(self.linear_2(x))
        x = self.bn_fc_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        
        return self.sigmoid(x)* np.pi*2
    
    ### network_9 with convolutionas, BIG BOY
    

    

class network_9(nn.Module): #do not work on cpu
    def __init__(self, input_size, n, output_size):
        super(network_9, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(93696, n) # change 76 to scalable version
        self.linear_2 = nn.Linear(n, n)
        self.linear_3 = nn.Linear(n, output_size)
        
        #convolutions
        
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                                  out_channels=108,
                                  kernel_size=11,
                                  stride=1,
                                  padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=108,
                                  out_channels=216,
                                  kernel_size=7,
                                  stride=1,
                                  padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=216,
                                  out_channels=512,
                                  kernel_size=5,
                                  stride=1,
                                  padding=1)
        self.conv1d_4 = nn.Conv1d(in_channels=512,
                                  out_channels=512,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)



        
        
        self.avg_pool1d_1 = nn.AvgPool1d(kernel_size=3,
                                         stride=None,
                                         padding=0)
        
        
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace = True)
        self.sigmoid = nn.Sigmoid()
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bn_cv_1 = nn.BatchNorm1d(108)
        self.bn_cv_2 = nn.BatchNorm1d(216)
        self.bn_cv_3 = nn.BatchNorm1d(512)
        self.bn_cv_4 = nn.BatchNorm1d(512)
        self.bn_fc_1 = nn.BatchNorm1d(n) #wont work on cpu

        self.dropout = nn.Dropout(0.1)
        

    def forward(self,x):
        x = torch.unsqueeze(x, 1)
        #print(x.shape)
        x = self.conv1d_1(x)
        x = self.avg_pool1d_1(x)
        self.bn_cv_1(x)
        
        x = self.conv1d_2(x)
        x = self.avg_pool1d_1(x)
        self.bn_cv_2(x)
        
        x = self.conv1d_3(x)  
        x = self.avg_pool1d_1(x)
        self.bn_cv_3(x)
        
        x = self.conv1d_4(x)
        self.bn_cv_4(x)
        
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.elu(self.linear_1(x))
        x = self.dropout(x)
        x = self.elu(self.linear_2(x))
        x = self.bn_fc_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        
        x = torch.squeeze(x)
        return self.sigmoid(x)* np.pi*2
