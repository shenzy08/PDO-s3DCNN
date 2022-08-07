from math import *
import numpy as np
import scipy
from scipy import sparse, linalg
from conf import settings
from group_representation import group_rep
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import os
from SO3_batchnorm import SO3_BatchNorm
import torch.utils.checkpoint
import copy
from discretized_PDO import discretized_PDO

def null(A, eps = 1e-10):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

class Cap_Type(object):
    class Struct(object):
        def __init__(self, rep, mul):
            # "rep" denotes the name of group_representation
            # "N" and "M" are used when rep is Cyclic
            # mul_list is the multiplicity of representation, only one integer when rep is not SO(2) or SO3
            # order_list are used when rep is SO2 or SO3
            self.rep = rep
            self.mul = mul
    def get_type(self, rep, mul):
        return self.Struct(rep, mul)
    
        
class ScalarActivation(torch.nn.Module):
    def __init__(self, enable, bias=True, inplace=False):
        '''
        Can be used only with scalar fields

        :param enable: list of tuple (dimension, activation function or None)
        :param bool bias: add a bias before the applying the activation
        '''
        super().__init__()

        self.inplace = inplace
        self.enable = []
        for d, act in enable:
            if d == 0:
                continue

            if self.enable and self.enable[-1][1] is act:
                self.enable[-1] = (self.enable[-1][0] + d, act)
            else:
                self.enable.append((d, act))

        nbias = sum([d for d, act in self.enable if act is not None])
        if bias and nbias > 0:
            self.bias = torch.nn.Parameter(torch.zeros(nbias))
        else:
            self.bias = None

    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [batch, feature, x, y, z]
        '''
        begin1 = 0
        begin2 = 0

        if self.inplace:
            output = input
        else:
            output = torch.empty_like(input)

        for d, act in self.enable:
            x = input[:, begin1:begin1 + d]

            if act is not None:
                if self.bias is not None:
                    x = x + self.bias[begin2:begin2 + d].view(1, -1, 1, 1, 1)
                    begin2 += d

                x = act(x)

            if not self.inplace or act is not None:
                output[:, begin1:begin1 + d] = x

            begin1 += d

        assert begin1 == input.size(1)
        assert self.bias is None or begin2 == self.bias.size(0)

        return output        

class SO3_layer(nn.Module):
    def __init__(self, group, in_type_list, out_type_list,conv_stride=1,discretized_mode='FD',use_BN=False,activation=(F.relu,torch.sigmoid),bias=True,checkpoint=True,capsule_dropout_p=None):
        super(SO3_layer, self).__init__()

        # activation
        if type(activation) is tuple:
            scalar_activation, gate_activation = activation
        else:
            scalar_activation, gate_activation = activation, activation

        repr_out = []
        for cur_type in out_type_list:
            repr_out.append(cur_type.mul)
            
        repr_out = tuple(repr_out)
        self.repr_out = repr_out

        Rs_out_with_gate = [(m, l) for l, m in enumerate(repr_out)]
        
        if (scalar_activation is not None and repr_out[0] > 0):
            self.scalar_act = ScalarActivation([(repr_out[0], scalar_activation)], bias=False)
        else:
            self.scalar_act = None

        self.out_type_list_with_gate = copy.deepcopy(out_type_list)
        cap_type = Cap_Type()
        
        n_non_scalar = sum(repr_out[1:])
        if gate_activation is not None and n_non_scalar > 0:
            Rs_out_with_gate.append((n_non_scalar, 0))  # concatenate scalar gate capsules after normal capsules
            self.out_type_list_with_gate.append(cap_type.get_type("trivial",n_non_scalar))
            self.gate_act = ScalarActivation([(n_non_scalar, gate_activation)], bias=bias)
        else:
            self.gate_act = None

                       
        # conv params
        self.conv_stride = conv_stride
#         self.smooth_stride = smooth_stride
        if discretized_mode == 'Gaussian':
            total_partial = torch.tensor(discretized_PDO(mode='Gaussian',r=2,sigma=1.),dtype=torch.float32)
        elif discretized_mode == 'FD':
            total_partial = torch.tensor(discretized_PDO(mode='FD'),dtype=torch.float32)

        total_partial = torch.tensor(total_partial,dtype=torch.float32)  
        self.size = total_partial.shape[1]
        self.padding = self.size//2

        self.register_buffer("total_partial",total_partial)
        
        self.group = group
        self.in_type_list = in_type_list
                
        self.in_channel = sum([in_type.mul for in_type in self.in_type_list])
        self.out_channel = sum([out_type.mul for out_type in self.out_type_list_with_gate])

        for i, out_type in enumerate(self.out_type_list_with_gate):
            out_mul = out_type.mul
            out_B0_shape_list = []
            out_B1_shape_list = []
            out_B2_shape_list = []

            for j, in_type in enumerate(in_type_list):
                in_mul = in_type.mul
                if not os.path.exists(self.group+"_base"):
                    os.mkdir(self.group+"_base")
                
                base_dir = "_to_".join([in_type.rep,out_type.rep])
                                    
                if os.path.exists(self.group +"_base/"+base_dir):

                    B0 = np.load(self.group +"_base/"+base_dir+"/B0.npy")
                    B1 = np.load(self.group +"_base/"+base_dir+"/B1.npy")
                    B2 = np.load(self.group +"_base/"+base_dir+"/B2.npy")
                else:
                    os.mkdir(self.group +"_base/"+base_dir)
                    B0, B1, B2 = self.get_B(in_type,out_type)
                    np.save(self.group +"_base/"+base_dir+"/B0.npy", B0)
                    np.save(self.group +"_base/"+base_dir+"/B1.npy", B1)
                    np.save(self.group +"_base/"+base_dir+"/B2.npy", B2)                                              
          
                if B0.shape[-1] > 0:
                    B0 = torch.tensor(B0,dtype=torch.float32) # (out_dim, 1, in_dim, num_B0)
                    B0_base = torch.einsum('ijkl,jmno->iklmno',B0,self.total_partial[:1]) # (out_dim,in_dim,num_B0,3,3,3)
                    out_dim, _, in_dim = B0.shape[:3]
                    B0_base_norm = B0_base.reshape(out_dim,in_dim,-1).norm(dim=-1).mean()
                    B0_base = B0_base/B0_base_norm
                else:
                    B0_base = torch.tensor(())
                
                if B1.shape[-1] > 0:
                    B1 = torch.tensor(B1,dtype=torch.float32)
                    out_dim, _, in_dim = B1.shape[:3]
                    B1_base = torch.einsum('ijkl,jmno->iklmno',B1,self.total_partial[1:4]) # (out_dim,in_dim,num_B0,3,3,3)
                    B1_base_norm = B1_base.reshape(out_dim,in_dim,-1).norm(dim=-1).mean()
                    B1_base = B1_base/B1_base_norm
                else:
                    B1_base = torch.tensor(())
                
                if B2.shape[-1] > 0:
                    
                    B2 = torch.tensor(B2,dtype=torch.float32)
                    out_dim, _, in_dim = B2.shape[:3]
                    B2_base = torch.einsum('ijkl,jmno->iklmno',B2,self.total_partial[4:]) # (out_dim,in_dim,num_B0,3,3,3)
                    B2_base_norm = B2_base.reshape(out_dim,in_dim,-1).norm(dim=-1).mean()
                    B2_base = B2_base/B2_base_norm

                else:
                    B2_base = torch.tensor(())
                self.register_buffer("B_base_{}".format(base_dir),torch.cat((B0_base,B1_base,B2_base),dim=2))
                
                
#                 initialization
                base_num = B0.shape[-1]+B1.shape[-1]+B2.shape[-1]
                stdv = sqrt(2./self.out_channel/base_num) 
                setattr(self,"weights_{}".format(base_dir),Parameter(stdv*torch.randn((out_mul,in_mul,base_num),dtype=torch.float32)))

        null_tensor = torch.tensor(())
        self.register_buffer("null_tensor",null_tensor)
        
        # BN params
        self.use_BN = use_BN
        if self.use_BN:
            self.bn = SO3_BatchNorm(self.out_type_list_with_gate)
        
        # dropout
        self.dropout = None
        if capsule_dropout_p is not None:
            Rs_out_without_gate = [(mul, 2 * n + 1) for n, mul in enumerate(repr_out)]  # Rs_out without gates
            self.dropout = SE3Dropout(Rs_out_without_gate, capsule_dropout_p)

        self.checkpoint = checkpoint
    
    def get_B(self,in_type,out_type):

        if self.group in ["SO2"]:
            g, rho_in, dim_in = group_rep(self.group, in_type)
            _, rho_out, dim_out = group_rep(self.group, out_type)

            coef_matrix0 = np.kron(rho_in.T,np.eye(dim_out)) - np.kron(np.eye(dim_in), rho_out) 
            V_B0 = null(coef_matrix0)
            coef_matrix1 = np.kron(np.kron(g,rho_in).T,np.eye(dim_out)) - np.kron(np.eye(3*dim_in), rho_out) 
            V_B1 = null(coef_matrix1)
            coef_matrix2 = np.kron(np.kron(settings.P_inv.dot(np.kron(g,g)).dot(settings.P),rho_in).T,np.eye(dim_out)) - np.kron(np.eye(6*dim_in), rho_out) 
            V_B2 = null(coef_matrix2)

        elif self.group in ["SO3"]:
            g1, g2, rho_in1, rho_in2, dim_in = group_rep(self.group, in_type)
            _, _, rho_out1, rho_out2, dim_out = group_rep(self.group, out_type)

            coef_matrix01 = np.kron(rho_in1.T,np.eye(dim_out)) - np.kron(np.eye(dim_in), rho_out1)
            coef_matrix02 = np.kron(rho_in2.T,np.eye(dim_out)) - np.kron(np.eye(dim_in), rho_out2) 
            coef_matrix0 = np.concatenate((coef_matrix01,coef_matrix02), axis=0)
            V_B0 = null(coef_matrix0)

            coef_matrix11 = np.kron(np.kron(g1,rho_in1).T,np.eye(dim_out)) - np.kron(np.eye(3*dim_in), rho_out1) 
            coef_matrix12 = np.kron(np.kron(g2,rho_in2).T,np.eye(dim_out)) - np.kron(np.eye(3*dim_in), rho_out2) 
            coef_matrix1 = np.concatenate((coef_matrix11,coef_matrix12), axis=0)
            V_B1 = null(coef_matrix1)

            coef_matrix21 = np.kron(np.kron(settings.P_inv.dot(np.kron(g1,g1)).dot(settings.P),rho_in1).T,np.eye(dim_out)) - np.kron(np.eye(6*dim_in), rho_out1) 
            coef_matrix22 = np.kron(np.kron(settings.P_inv.dot(np.kron(g2,g2)).dot(settings.P),rho_in2).T,np.eye(dim_out)) - np.kron(np.eye(6*dim_in), rho_out2) 
            coef_matrix2 = np.concatenate((coef_matrix21,coef_matrix22), axis=0)
            V_B2 = null(coef_matrix2)

        else:
            raise AssertionError("Wrong group!")

        B0 = V_B0.reshape(1,dim_in,dim_out,-1).transpose(2,0,1,3)
        B1 = V_B1.reshape(3,dim_in,dim_out,-1).transpose(2,0,1,3)
        B2 = V_B2.reshape(6,dim_in,dim_out,-1).transpose(2,0,1,3)

        
        return B0, B1, B2
            
    def forward(self, ipt):        
        def conv(ipt):
            total_weight = self.null_tensor
            for out_type in self.out_type_list_with_gate:
                out_type_weight = self.null_tensor
                for in_type in self.in_type_list:
                    base_dir = "_to_".join([in_type.rep,out_type.rep])
                    weight_B = getattr(self,"weights_{}".format(base_dir)) # (out_mul,in_mul,10) (i,j,k)
                    out_mul, in_mul, _ = weight_B.shape
                    B_base = getattr(self,"B_base_{}".format(base_dir)) # (out_dim,in_dim,10,3,3,3) (l,m,k,n,o,p)
                    out_dim, in_dim = B_base.shape[:2]
                    weight = torch.einsum("ijk,lmknop->iljmnop",weight_B,B_base).reshape(out_mul*out_dim,in_mul*in_dim,self.size,self.size,self.size) #

                    out_type_weight = torch.cat((out_type_weight,weight),dim=1)
                total_weight = torch.cat((total_weight,out_type_weight),dim=0) # (out_channel, in_channel)
            opt = nn.functional.conv3d(ipt,total_weight,stride=self.conv_stride,padding=self.padding)
            return opt
            
        def gate(y):                
            nbatch = y.size(0)
            nx = y.size(2)
            ny = y.size(3)
            nz = y.size(4)

            size_out = sum(mul * (2 * n + 1) for n, mul in enumerate(self.repr_out))

            if self.gate_act is not None:
                g = y[:, size_out:]
                g = self.gate_act(g)
                begin_g = 0  # index of first scalar gate capsule

            z = y.new_empty((y.size(0), size_out, y.size(2), y.size(3), y.size(4)))
            begin_y = 0  # index of first capsule

            for n, mul in enumerate(self.repr_out):
                if mul == 0:
                    continue
                dim = 2 * n + 1

                # crop out capsules of order n
                field_y = y[:, begin_y: begin_y + mul * dim]  # [batch, feature * repr, x, y, z]

                if n == 0:
                    # Scalar activation
                    if self.scalar_act is not None:
                        field = self.scalar_act(field_y)
                    else:
                        field = field_y
                else:
                    if self.gate_act is not None:
                        # reshape channels in capsules and capsule entries
                        field_y = field_y.contiguous()
                        field_y = field_y.view(nbatch, mul, dim, nx, ny, nz)  # [batch, feature, repr, x, y, z]

                        # crop out corresponding scalar gates
                        field_g = g[:, begin_g: begin_g + mul]  # [batch, feature, x, y, z]
                        begin_g += mul
                        # reshape channels for broadcasting
                        field_g = field_g.contiguous()
                        field_g = field_g.view(nbatch, mul, 1, nx, ny, nz)  # [batch, feature, repr, x, y, z]

                        # scale non-scalar capsules by gate values
                        field = field_y * field_g  # [batch, feature, repr, x, y, z]
                        field = field.view(nbatch, mul * dim, nx, ny, nz)  # [batch, feature * repr, x, y, z]
                        del field_g
                    else:
                        field = field_y
                del field_y

                z[:, begin_y: begin_y + mul * dim] = field
                begin_y += mul * dim
                del field

            return z
            
        opt = conv(ipt)
        # bn
        if self.use_BN:
            opt = self.bn(opt)  
            
        # gate
        if self.scalar_act is not None or self.gate_act is not None:
            opt = torch.utils.checkpoint.checkpoint(gate, opt) if self.checkpoint else gate(opt)
            
        # smooth_stride
#         if self.smooth_stride:
#             opt = low_pass_filter(opt,2,2)
            
               # dropout
        if self.dropout is not None:
            opt = self.dropout(opt)
            
        return opt

def low_pass_filter(image, scale, stride=1):
    """
    :param tensor image: [..., x, y, z]
    :param float scale: 
    :param int stride:
    """
    if scale <= 1:
        return image

    sigma = 0.5 * (scale ** 2 - 1) ** 0.5

    size = int(1 + 2 * 2.5 * sigma)
    if size % 2 == 0:
        size += 1

    rng = torch.arange(size, dtype=image.dtype, device=image.device) - size // 2  # [-(size // 2), ..., size // 2]
    x = rng.view(size, 1, 1).expand(size, size, size)
    y = rng.view(1, size, 1).expand(size, size, size)
    z = rng.view(1, 1, size).expand(size, size, size)

    kernel = torch.exp(- (x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size, size, size)

    out = F.conv3d(image.view(-1, 1, *image.size()[-3:]), kernel, padding=size // 2, stride=stride)
    out = out.view(*image.size()[:-3], *out.size()[-3:])

    return out





