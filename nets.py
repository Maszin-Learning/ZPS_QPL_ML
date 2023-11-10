import torch.nn as nn
import torch


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
        #print(x.shape)
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
        #print(x.shape)
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
    
### network_5 4 with relu
class network_5(nn.Module):
    def __init__(self, input_size, n, output_size):
        super(network_5, self).__init__()
        self.input = input_size
        self.output = output_size

        self.linear_1 = nn.Linear(input_size, n)
        self.linear_2 = nn.Linear(n, n)
        self.linear_3 = nn.Linear(n, output_size)
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace = True)
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(n) #wont work on cpu
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        #print(x.shape)
        x = self.relu(self.linear_1(x))
        x = self.bn_1(x)
        x = self.relu(self.linear_2(x))
        x = self.bn_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        return x
    
### network_6 4 with tanh
class network_6(nn.Module):
    def __init__(self, input_size, n, output_size):
        super(network_6, self).__init__()
        self.input = input_size
        self.output = output_size

        self.linear_1 = nn.Linear(input_size, n)
        self.linear_2 = nn.Linear(n, n)
        self.linear_3 = nn.Linear(n, output_size)
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace = True)
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(n) #wont work on cpu
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        #print(x.shape)
        x = self.tanh(self.linear_1(x))
        x = self.bn_1(x)
        x = self.tanh(self.linear_2(x))
        x = self.bn_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        return x
    
    ### network_7 with convolutionas
class network_7(nn.Module): #do not work on cpu
    def __init__(self, input_size, n, output_size):
        super(network_7, self).__init__()
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
        
        return x
    

    
    
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
        
        return x
    
    ### network_9 with convolutionas, BIG BOY
class network_9(nn.Module): #do not work on cpu
    def __init__(self, input_size, n, output_size):
        super(network_9, self).__init__()
        self.input = input_size
        self.output = output_size
        self.linear_1 = nn.Linear(93696, n) # change 76 to scalable wersion
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
        
        self.normal_1 = nn.LayerNorm(n)
        self.normal_3 = nn.LayerNorm(output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bn_fc_1 = nn.BatchNorm1d(n) #wont work on cpu
        self.bn_fc_1 = nn.BatchNorm1d(n)
        self.dropout = nn.Dropout(0.1)
        

    def forward(self,x):
        x = torch.unsqueeze(x, 1)
        #print(x.shape)
        x = self.conv1d_1(x)
        x = self.avg_pool1d_1(x)
        x = self.conv1d_2(x)
        x = self.avg_pool1d_1(x)
        x = self.conv1d_3(x)  
        #x = self.dropout(x)
        x = self.avg_pool1d_1(x)
        x = self.conv1d_4(x)
        #x = self.dropout(x)
        #print(x.shape) 
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        #print(x.shape)
        x = self.elu(self.linear_1(x))
        x = self.dropout(x)
        x = self.elu(self.linear_2(x))
        x = self.bn_fc_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        
        return x
    
