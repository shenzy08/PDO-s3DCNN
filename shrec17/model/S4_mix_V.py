import torch
from torch import nn
import sys
sys.path.append("...")
from util import PDO_e3DConv, low_pass_filter, Cap_Type, G_BN
from group_representation import group_rep
import datetime
import torch.nn.functional as F

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

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        group = "S4"
        cap_type = Cap_Type()
        layer_type_list = [[cap_type.get_type("trivial", 1)],
                  [cap_type.get_type("regular", 4),cap_type.get_type("quotient_V", 2)],
                  [cap_type.get_type("regular", 4),cap_type.get_type("quotient_V", 2)],
                  [cap_type.get_type("regular", 6),cap_type.get_type("quotient_V", 3)],
                  [cap_type.get_type("regular", 6),cap_type.get_type("quotient_V", 3)],
                  [cap_type.get_type("regular", 8),cap_type.get_type("quotient_V", 3)],
                  [cap_type.get_type("regular", 8),cap_type.get_type("quotient_V", 4)],
                  [cap_type.get_type("regular", 8),cap_type.get_type("quotient_V", 4)],
                  [cap_type.get_type("trivial", 512)]]

        self.conv1 = PDO_e3DConv(group, layer_type_list[0], layer_type_list[1],stride=2)
        self.bn1 = G_BN(group,layer_type_list[1])
        self.conv2 = PDO_e3DConv(group, layer_type_list[1], layer_type_list[2])
        self.bn2 = G_BN(group,layer_type_list[2])
        self.conv3 = PDO_e3DConv(group, layer_type_list[2], layer_type_list[3])
        self.bn3 = G_BN(group,layer_type_list[3])
        self.conv4 = PDO_e3DConv(group, layer_type_list[3], layer_type_list[4])
        self.bn4 = G_BN(group,layer_type_list[4])
        self.conv5 = PDO_e3DConv(group, layer_type_list[4], layer_type_list[5])
        self.bn5 = G_BN(group,layer_type_list[5])
        self.conv6 = PDO_e3DConv(group, layer_type_list[5], layer_type_list[6])
        self.bn6 = G_BN(group,layer_type_list[6])
        self.conv7 = PDO_e3DConv(group, layer_type_list[6], layer_type_list[7])
        self.bn7 = G_BN(group,layer_type_list[7])
        self.conv8 = PDO_e3DConv(group, layer_type_list[7], layer_type_list[8])
        self.bn8 = G_BN(group,layer_type_list[8])
        
        self.avg = nn.AvgPool3d(2,2)
        self.relu = nn.ReLU(inplace=True)        
        self.AvgSpacial = AvgSpacial()
        self.linear = nn.Linear(512,55)


    def forward(self, inp):  # pylint: disable=W
        output = self.relu(self.bn1(self.conv1(inp)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.relu(self.bn3(self.conv3(output)))
        output = self.avg(output)
        output = self.relu(self.bn4(self.conv4(output)))
        output = self.relu(self.bn5(self.conv5(output)))
        output = self.avg(output)
        output = self.relu(self.bn6(self.conv6(output)))
        output = self.relu(self.bn7(self.conv7(output)))
        output = self.relu(self.bn8(self.conv8(output)))
        pool = self.AvgSpacial(output)
        output = self.linear(pool)
        
        return output



