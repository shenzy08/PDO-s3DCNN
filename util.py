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
            # self.order_list = order_list
    def get_type(self, rep, mul):
        return self.Struct(rep, mul)
    
class G_BN(nn.Module):
    def __init__(self, group, type_list):
        super(G_BN, self).__init__()
        
        self.group = group
        self.type_list = type_list
        self.BN_list = nn.ModuleList()
        for cur_type in self.type_list:
            mul = cur_type.mul
            if self.group in ["Cyclic","V","A4","S4","A5"]:
                self.BN_list.append(nn.BatchNorm3d(mul))
                
            elif self.group in ["SO2"]:
                pass
            elif self.group in ["SO3"]:
                pass
      
    def forward(self,ipt):
        opt_list = []
        start = 0
        for i, cur_type in enumerate(self.type_list):
            mul = cur_type.mul
            dim = group_rep(self.group, cur_type)[-1]
            # discrete group
            if self.group in ["Cyclic","V","A4","S4","A5"]:
                tmp = ipt[:,start:start+mul*dim,...].reshape(ipt.shape[0],mul,ipt.shape[2]*dim,ipt.shape[3],ipt.shape[4])
                tmp = self.BN_list[i](tmp)
                tmp = tmp.reshape(ipt.shape[0],mul*dim,ipt.shape[2],ipt.shape[3],ipt.shape[4])
                opt_list.append(tmp)
                
            elif self.group in ["SO2"]:
                pass
            elif self.group in ["SO3"]:
                pass       
            start += mul * dim
        opt = torch.cat(opt_list,1)
        return opt
        
        

class PDO_e3DConv(nn.Module):
    def __init__(self, group, in_type_list, out_type_list,stride=1,discretized_mode='FD',device=None):
        super(PDO_e3DConv, self).__init__()

        self.stride = stride
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
        self.out_type_list = out_type_list
        
        self.in_channel = sum([in_type.mul for in_type in self.in_type_list])
        self.out_channel = sum([out_type.mul for out_type in self.out_type_list])
        self.discretized_mode = discretized_mode

        for i, out_type in enumerate(out_type_list):
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
                    B0 = torch.tensor(B0,dtype=torch.float32)
                    B0_base = torch.einsum('ijkl,jmno->iklmno',B0,self.total_partial[:1]) # (out_dim,in_dim,num_B0,3,3,3)
                    out_dim, _, in_dim, _ = B0.shape
                    B0_base_norm = B0_base.reshape(out_dim,in_dim,-1).norm(dim=-1).mean()
                    B0_base = B0_base/B0_base_norm
                else:
                    B0_base = torch.tensor(())
                
                if B1.shape[-1] > 0:
                    B1 = torch.tensor(B1,dtype=torch.float32)
                    out_dim, _, in_dim, _ = B1.shape
                    B1_base = torch.einsum('ijkl,jmno->iklmno',B1,self.total_partial[1:4]) # (out_dim,in_dim,num_B0,3,3,3)
                    B1_base_norm = B1_base.reshape(out_dim,in_dim,-1).norm(dim=-1).mean()
                    B1_base = B1_base/B1_base_norm
                else:
                    B1_base = torch.tensor(())
                
                if B2.shape[-1] > 0:
                    B2 = torch.tensor(B2,dtype=torch.float32)
                    out_dim, _, in_dim, _ = B2.shape
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
                    
    def get_B(self,in_type,out_type):
        
        if self.group in ["Cyclic", "SO2"]:
            g, rho_in, dim_in = group_rep(self.group, in_type)
            _, rho_out, dim_out = group_rep(self.group, out_type)

            coef_matrix0 = np.kron(rho_in.T,np.eye(dim_out)) - np.kron(np.eye(dim_in), rho_out) 
            V_B0 = null(coef_matrix0)
            coef_matrix1 = np.kron(np.kron(g,rho_in).T,np.eye(dim_out)) - np.kron(np.eye(3*dim_in), rho_out) 
            V_B1 = null(coef_matrix1)
            coef_matrix2 = np.kron(np.kron(settings.P_inv.dot(np.kron(g,g)).dot(settings.P),rho_in).T,np.eye(dim_out)) - np.kron(np.eye(6*dim_in), rho_out) 
            V_B2 = null(coef_matrix2)

        elif self.group in ["V","A4","S4","A5","SO3"]:
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

            for out_type in self.out_type_list:
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

            opt = nn.functional.conv3d(ipt,total_weight,stride=self.stride,padding=self.padding)
            return opt

        opt = conv(ipt)
        
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

def rotate_scalar(x, rot):
    # this function works in the convention field[x, y, z]
    invrot = np.linalg.inv(rot)
    center = (np.array(x.shape) - 1) / 2
    return affine_transform(x, matrix=invrot, offset=center - np.dot(invrot, center))


def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, dtype=torch.get_default_dtype())
    return gamma.new_tensor([
        [gamma.cos(), -gamma.sin(), 0],
        [gamma.sin(), gamma.cos(), 0],
        [0, 0, 1]
    ])


def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, dtype=torch.get_default_dtype())
    return beta.new_tensor([
        [beta.cos(), 0, beta.sin()],
        [0, 1, 0],
        [-beta.sin(), 0, beta.cos()]
    ])


def rot(alpha, beta, gamma):
    '''
    ZYZ Eurler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)



