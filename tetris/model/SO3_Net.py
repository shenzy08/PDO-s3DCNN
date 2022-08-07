import torch
from torch import nn
import sys
sys.path.append("...")
from util_SO3 import SO3_layer, low_pass_filter, Cap_Type
from group_representation import group_rep
import datetime
import torch.nn.functional as F
from group_representation import group_rep

class low_pass_filter:
    def __init__(self, size, sigma=1.3, padding=1, stride=2):

        self.padding = padding
        self.stride = stride

        def gaussian_filter(size,sigma):
#             print(type(sigma))

            rng = torch.arange(size)  # [0, ..., size-1]
            x = rng.view(size, 1, 1).expand(size, size, size)
            y = rng.view(1, size, 1).expand(size, size, size)
            z = rng.view(1, 1, size).expand(size, size, size)

            center_x = (size-1)/2.
            center_y = (size-1)/2.
            center_z = (size-1)/2.

            kernel = torch.exp(- ((x-center_x) ** 2 + (y-center_y) ** 2 + (z-center_z) ** 2)/2/sigma**2)
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, size, size, size)

            return kernel

        self.kernel = gaussian_filter(size,sigma)
       
    def __call__(self,x):
        
        kernel = self.kernel.repeat(x.shape[1],1,1,1,1)
        if x.is_cuda:
            kernel = kernel.cuda()
        out = F.conv3d(x,kernel,stride=self.stride,padding=self.padding,groups=x.shape[1])

        return out

class AvgSpacial(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)


class SO3_Net(nn.Module):

    def __init__(self,discretized_mode,drop_rate):
        super().__init__()

        group = "SO3"
        cap_type = Cap_Type()
        layer_type_list = [[cap_type.get_type("trivial", 1)],
                          [cap_type.get_type("irrep0", 4),cap_type.get_type("irrep1", 4),cap_type.get_type("irrep2", 4)],
                          [cap_type.get_type("irrep0", 16),cap_type.get_type("irrep1", 16),cap_type.get_type("irrep2", 16)],
                          [cap_type.get_type("irrep0", 32),cap_type.get_type("irrep1", 16),cap_type.get_type("irrep2", 16)]]
                  
        self.layer1 = SO3_layer(group, layer_type_list[0], layer_type_list[1],discretized_mode=discretized_mode,activation=(F.relu,torch.sigmoid))
        self.layer2 = SO3_layer(group, layer_type_list[1], layer_type_list[2],discretized_mode=discretized_mode,activation=(F.relu,torch.sigmoid))
        self.layer3 = SO3_layer(group, layer_type_list[2], layer_type_list[3],discretized_mode=discretized_mode,activation=(F.relu,torch.sigmoid))
        self.smooth = low_pass_filter(size=5,sigma = 1., padding = 2,stride=1)        
        layer_type_list.append([cap_type.get_type("trivial", 128)])
        self.layer4 = SO3_layer(group, layer_type_list[3], layer_type_list[4],discretized_mode=discretized_mode,activation=(F.relu,torch.sigmoid))            
        
        if discretized_mode == 'FD':
            self.pool = nn.AvgPool3d(2,2)
        elif discretized_mode == 'Gaussian':
            self.pool = low_pass_filter(size=4,sigma =1.,padding = 1,stride = 2)
        
        self.AvgSpacial = AvgSpacial()
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(128,8)

    def forward(self, inp):  # pylint: disable=W
        inp = self.smooth(inp)
        output = self.layer1(inp)
        output = self.layer2(output)
        output = self.pool(output)
        output = self.layer3(output)
        output = self.pool(output)
        output = self.layer4(output)
        pool = self.AvgSpacial(output)
        pool = self.dropout(pool)
        output = self.linear(pool)
        
        return output


