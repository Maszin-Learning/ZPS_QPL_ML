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
        self.linear_3 = nn.Linear(240,output_size)
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


#UNET
class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, padding = 3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
        return out       

class se_block(nn.Module):
    def __init__(self,in_layer, out_layer):
        super(se_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)
        return x_out

class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()
        
        self.cbr1 = conbr_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)
    
    def forward(self,x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out          

class UNET_1D(nn.Module):
    def __init__(self ,input_dim,layer_n,kernel_size,depth):
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        
        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)
        
        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,5, 2)
        self.layer3 = self.down_layer(int(self.layer_n*2)+int(self.input_dim), int(self.layer_n*3), self.kernel_size,5, 2)
        self.layer4 = self.down_layer(int(self.layer_n*3)+int(self.input_dim), int(self.layer_n*4), self.kernel_size,5, 2)
        self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size,4, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')
        
        self.outcov = nn.Conv1d(self.layer_n, 11, kernel_size=self.kernel_size, stride=1,padding = 3)
    
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
            
    def forward(self, x):
        
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        
        x = torch.cat([out_2,pool_x2],1)
        x = self.layer4(x)
        
        #############Decoder####################
        
        up = self.upsample1(x)
        up = torch.cat([up,out_2],1)
        up = self.cbr_up1(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_1],1)
        up = self.cbr_up2(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_0],1)
        up = self.cbr_up3(up)
        
        out = self.outcov(up)
        
        #out = nn.functional.softmax(out,dim=2)
        
        return out