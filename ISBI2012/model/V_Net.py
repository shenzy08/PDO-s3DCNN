import torch
from torch import nn
import sys
sys.path.append("...")
sys.path.append("..")
from util import PDO_e3DConv, low_pass_filter, Cap_Type, G_BN
from group_representation import group_rep
import datetime
import torch.nn.functional as F
from Basic_blocks import * 

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

    def __init__(self, out_dim, final_act, dropout_rate, pooling_mode, num_fc):
        super().__init__()

        self.act_fn = nn.LeakyReLU(0.2, inplace=True)
        self.act_fn_2 = nn.ReLU()
        self.num_fc = num_fc

        print("\n------Initiating PDO-e3DCNN------\n")

        self.group = "V"
        self.out_dim = out_dim
        cap_type = Cap_Type()
        self.layer_type_list = [[cap_type.get_type("trivial", 1)],
                  [cap_type.get_type("regular", 20)],
                  [cap_type.get_type("regular", 40)],
                  [cap_type.get_type("regular", 80)],
                  [cap_type.get_type("regular", 160)],
                  [cap_type.get_type("regular", 240)],
                  [cap_type.get_type("trivial", self.out_dim)]]

        # encoder

        self.down_1 = Conv_residual_conv(self.group, self.layer_type_list[0], self.layer_type_list[1], self.act_fn, dropout_rate)
        self.pool_1 = pool(pooling_mode)
        self.down_2 = Conv_residual_conv(self.group, self.layer_type_list[1], self.layer_type_list[2], self.act_fn, dropout_rate)
        self.pool_2 = pool(pooling_mode)
        self.down_3 = Conv_residual_conv(self.group, self.layer_type_list[2], self.layer_type_list[3], self.act_fn, dropout_rate)
        self.pool_3 = pool(pooling_mode)
        self.down_4 = Conv_residual_conv(self.group, self.layer_type_list[3], self.layer_type_list[4], self.act_fn, dropout_rate)
        self.pool_4 = pool(pooling_mode)

        # bridge

        self.bridge = Conv_residual_conv(self.group, self.layer_type_list[4], self.layer_type_list[5], self.act_fn, dropout_rate)

        # decoder

        self.deconv_1 = conv_trans_block(self.group, self.layer_type_list[5], self.layer_type_list[4], self.act_fn_2, dropout_rate)
        self.up_1 = Conv_residual_conv(self.group, self.layer_type_list[4], self.layer_type_list[4], self.act_fn_2, dropout_rate)
        self.deconv_2 = conv_trans_block(self.group, self.layer_type_list[4], self.layer_type_list[3], self.act_fn_2, dropout_rate)
        self.up_2 = Conv_residual_conv(self.group, self.layer_type_list[3], self.layer_type_list[3], self.act_fn_2, dropout_rate)
        self.deconv_3 = conv_trans_block(self.group, self.layer_type_list[3], self.layer_type_list[2], self.act_fn_2, dropout_rate)
        self.up_3 = Conv_residual_conv(self.group, self.layer_type_list[2], self.layer_type_list[2], self.act_fn_2, dropout_rate)
        self.deconv_4 = conv_trans_block(self.group, self.layer_type_list[2], self.layer_type_list[1], self.act_fn_2, dropout_rate)
        self.up_4 = Conv_residual_conv(self.group, self.layer_type_list[1], self.layer_type_list[1], self.act_fn_2, dropout_rate)

        # output

        self.out1 = conv_block(self.group,self.layer_type_list[1],self.layer_type_list[-1],self.act_fn_2,dropout_rate)
        self.out2 = plain_conv_block(self.out_dim,self.out_dim,self.act_fn_2)
        self.out3 = plain_conv_block(self.out_dim,self.out_dim,self.act_fn_2)
        self.out4 = nn.Conv3d(self.out_dim,1,kernel_size=1)
                                    
        self.final_act = final_act
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, inp):  # pylint: disable=W
        down_1 = self.down_1(inp)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4)/2
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3)/2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2)/2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1)/2
        up_4 = self.up_4(skip_4)

        out1 = self.out1(up_4)
        if self.num_fc == 1:
            out4 = self.out4(out1)
        elif self.num_fc == 2:
            out3 = self.out3(out1)
            out4 = self.out4(out3)
        elif self.num_fc == 3:
            out2 = self.out2(out1)
            out3 = self.out3(out2)
            out4 = self.out4(out3)
        
        if self.final_act == 'sigmoid':
            out = self.sigmoid(out4)
        if self.final_act == 'tanh':
            out = (self.tanh(out4)+1.)/2.

        return out




