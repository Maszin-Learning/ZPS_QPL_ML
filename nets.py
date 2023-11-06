import torch.nn as nn


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
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
        self.bn_1 = nn.BatchNorm1d(n) #wont work on cpu
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        x = self.leakyrelu(self.linear_1(x))
        x = self.bn_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        return x
    

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
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
        self.bn_1 = nn.BatchNorm1d(n) #wont work on cpu
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        x = self.leakyrelu(self.linear_1(x))
        #x = self.bn_1(x)
        #x = self.bn_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        return x
    

### network_3
class network_3(nn.Module):
    def __init__(self, input_size, n, output_size):
        super(network_3, self).__init__()
        self.input = input_size
        self.output = output_size

        self.linear_1 = nn.Linear(input_size, n)
        self.linear_2 = nn.Linear(n, n)
        self.linear_3 = nn.Linear(n, output_size)
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace = True)
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
        self.bn_1 = nn.BatchNorm1d(n) #wont work on cpu
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        x = self.leakyrelu(self.linear_1(x))
        x = self.bn_1(x)
        x = self.leakyrelu(self.linear_2(x))
        x = self.bn_1(x)
        x = self.leakyrelu(self.linear_2(x))
        x = self.bn_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        return x
    

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
        return x