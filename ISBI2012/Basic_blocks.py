import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys
sys.path.append("..")
from util import PDO_e3DConv, low_pass_filter, Cap_Type, G_BN
from util_SO3 import SO3_layer, low_pass_filter, Cap_Type


def plain_conv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def conv_block(group, in_type_list,out_type_list,act_fn,dropout_rate):
    if group == "SO3":
        model = SO3_layer(group, in_type_list, out_type_list,use_BN = True, activation=(nn.ReLU(inplace=True), torch.sigmoid))
    else:       
        model = nn.Sequential(
            PDO_e3DConv(group, in_type_list, out_type_list),
            G_BN(group,out_type_list),
            act_fn,
            nn.Dropout(p=dropout_rate), 
        )
    return model


def conv_trans_block(group,in_type_list,out_type_list,act_fn,dropout_rate):
    if group == "SO3":
        model = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2),mode="nearest",align_corners=False),
            SO3_layer(group, in_type_list, out_type_list,use_BN = True, activation=(nn.ReLU(inplace=True), torch.sigmoid))
        )
    else:
        model = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2),mode="nearest"),
            PDO_e3DConv(group, in_type_list, out_type_list),
            G_BN(group,out_type_list),
            act_fn,
            nn.Dropout(p=dropout_rate)
        )
    return model


def pool(pool_mode):
    if pool_mode == 'avg':
        pool = nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)
    elif pool_mode == 'max':
        pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)
    return pool


def conv_block_3(group,in_type_list,out_type_list,act_fn,dropout_rate):
    if group == "SO3":
        model = nn.Sequential(
            conv_block(group,in_type_list,out_type_list,act_fn,dropout_rate),
            conv_block(group,out_type_list,out_type_list,act_fn,dropout_rate),
            conv_block(group,out_type_list,out_type_list,act_fn,dropout_rate),
        )
    else:
        model = nn.Sequential(
            conv_block(group,in_type_list,out_type_list,act_fn,dropout_rate),
            conv_block(group,out_type_list,out_type_list,act_fn,dropout_rate),
            PDO_e3DConv(group, out_type_list, out_type_list),
            G_BN(group,out_type_list),
        )
    return model


class Conv_residual_conv(nn.Module):

    def __init__(self,group,in_type_list,out_type_list,act_fn,dropout_rate):
        super(Conv_residual_conv,self).__init__()
        self.in_type_list = in_type_list
        self.out_type_list = out_type_list
        self.group = group
        act_fn = act_fn

        self.conv_1 = conv_block(self.group,self.in_type_list,self.out_type_list,act_fn,dropout_rate)
        self.conv_2 = conv_block_3(self.group,self.out_type_list,self.out_type_list,act_fn,dropout_rate)
        self.conv_3 = conv_block(self.group,self.out_type_list,self.out_type_list,act_fn,dropout_rate)

    def forward(self,input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3