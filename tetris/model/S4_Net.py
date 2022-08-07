import torch
from torch import nn
import sys
sys.path.append("...")
from util import PDO_e3DConv, Cap_Type
import datetime

class AvgSpacial(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)

class LowPass(nn.Module):
    def __init__(self, scale, stride):
        super().__init__()
        self.scale = scale
        self.stride = stride

    def forward(self, inp):
        return low_pass_filter(inp, self.scale, self.stride)

class S4_Net(nn.Module):

    def __init__(self):
        super().__init__()

        group = "S4"
        cap_type = Cap_Type()
        layer_type_list = [[cap_type.get_type("trivial", 1)],
                          [cap_type.get_type("regular", 10)],
                          [cap_type.get_type("regular", 10)],
                          [cap_type.get_type("trivial", 64)]]

        self.conv1 = PDO_e3DConv(group, layer_type_list[0], layer_type_list[1])
        self.conv2 = PDO_e3DConv(group, layer_type_list[1], layer_type_list[2])
        self.conv3 = PDO_e3DConv(group, layer_type_list[2], layer_type_list[3])
        self.relu = nn.ReLU(inplace=True)
        self.avg = nn.AvgPool3d(kernel_size=2,stride=2,padding=0)
        self.AvgSpacial = AvgSpacial()
        self.linear = nn.Linear(64,8)

    def forward(self, inp):  # pylint: disable=W
        output = self.relu(self.conv1(inp))
        output = self.avg(output)
        output = self.relu(self.conv2(output))
        output = self.avg(output)
        output = self.relu(self.conv3(output))
        pool = self.AvgSpacial(output)
        output = self.linear(pool)
        
        return output
