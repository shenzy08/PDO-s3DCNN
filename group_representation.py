from math import *
import numpy as np
from wigner_D_matrix import wigner_D_matrix
from conf import settings
from scipy.sparse import coo_matrix

def group_rep(group, cap_type):
    rep = cap_type.rep
    if group[:-1] == "C_":
        N = group[-1]
        g = np.array([[cos(2*pi/N), -sin(2*pi)/N, 0],[sin(2*pi)/N, cos(2*pi)/N, 0],[0, 0, 1]])
        if rep == "trivial":
            rho = np.array([1])
        elif rep == "regular":
            ipt_capsule_dim = N
            data = [1] * ipt_capsule_dim
            row = list(range(1,ipt_capsule_dim)) + [0]
            col = list(range(ipt_capsule_dim))
            rho = coo_matrix((data,(row,col)),shape=(ipt_capsule_dim,ipt_capsule_dim)).toarray()
        elif rep[:-1] == "quotient_C_":
            M = int(rep[-1])
            if N % M != 0:
                raise AssertionError("N should mod M!")
            else:
                ipt_capsule_dim = N//M 
                data = [1] * ipt_capsule_dim
                row = list(range(1,ipt_capsule_dim)) + [0]
                col = list(range(ipt_capsule_dim))
                rho = coo_matrix((data,(row,col)),shape=(ipt_capsule_dim,ipt_capsule_dim)).toarray()                   
        else:
            raise AssertionError("Wrong group representations")
        return g, rho, rho.shape[0]
    if group[0] == "D":
        N = int(group[1:])
        gr = np.array([[cos(2*pi/N), -sin(2*pi/N), 0],[sin(2*pi/N), cos(2*pi/N), 0],[0, 0, 1]])
        gf = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])
        if rep == "trivial":
            ipt_capsule_dim = 1
            rhor = np.array([1])
            rhof = np.array([1])
        elif rep == "regular":
            ipt_capsule_dim = 2 * N
            rhor = np.zeros((2*N,2*N))
        #     print(rhor)
            rhor[0,N-1] = 1
            rhor[1:N,:N-1] = np.eye(N-1,N-1)
            rhor[-1,N] = 1
            rhor[N:2*N-1,N+1:] = np.eye(N-1,N-1)
            rhof = np.zeros((2*N,2*N))
            rhof[N:,:N] = np.eye(N,N)
            rhof[:N,N:] = np.eye(N,N)
        return gr, gf, rhor, rhof, rhor.shape[0]
       
    elif group == "V": 
        g2 = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]])
        g12 = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])
        if rep == "trivial":
            rho2 = np.array([1])
            rho12 = np.array([1])
        elif rep == "regular":
            ipt_capsule_dim = 4
            rho2 = np.array([[0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]])
            rho12 = np.array([[0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [1, 0, 0, 0],
                             [0, 1, 0, 0]])
        else:
            raise AssertionError("Wrong group representations")
        return g2, g12, rho2, rho12, rho2.shape[0]

    elif group == "A4":
        A4 = [0, 2, 7, 9, 10, 11, 12, 14, 19, 21, 22, 23]
        S4_cayley = np.array(settings.S4_cayley)
        A4_cayley = S4_cayley[A4][:,A4]
        # 生成元
        g2 = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]])
        g7 = np.array([[0, 0, 1],[1, 0, 0],[0, 1, 0]])
        if rep == "trivial":
            ipt_capsule_dim = 1
            rho2 = np.array([1])
            rho7 = np.array([1])
        elif rep == "regular":
            ipt_capsule_dim = len(A4)
            data = [1] * ipt_capsule_dim
            row2 = A4_cayley[1]
            row7 = A4_cayley[2]
            col = list(range(ipt_capsule_dim))
            rho2 = coo_matrix((data,(row2,col)),shape=(24,ipt_capsule_dim)).toarray()[A4]
            rho7 = coo_matrix((data,(row7,col)),shape=(24,ipt_capsule_dim)).toarray()[A4]
        elif rep == "quotient_V":
            ipt_capsule_dim = 3
            rho2 = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
            rho7 = np.array([[0, 0, 1],[1, 0, 0],[0, 1, 0]])                
        else:
            raise AssertionError("Wrong group representations")
        return g2, g7, rho2, rho7, rho2.shape[0]

    elif group == "S4":
        g1= np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]]) #p1
        g4 = np.array([[0, 0, 1],[0, 1, 0],[-1, 0, 0]]) #p4
        S4_cayley = settings.S4_cayley
        if rep == "trivial":
            ipt_capsule_dim = 1
            rho1 = np.array([1])
            rho4 = np.array([1])
        elif rep == "regular":
            ipt_capsule_dim = 24
            data = [1] * ipt_capsule_dim
            row1 = S4_cayley[1]
            row4 = S4_cayley[4]
            col = list(range(ipt_capsule_dim))
            rho1 = coo_matrix((data,(row1,col)),shape=(ipt_capsule_dim,ipt_capsule_dim)).toarray()
            rho4 = coo_matrix((data,(row4,col)),shape=(ipt_capsule_dim,ipt_capsule_dim)).toarray()
        elif rep == "quotient_A4":
            ipt_capsule_dim = 2
            rho1 = np.array([[0, 1],[1, 0]])
            rho4 = np.array([[0, 1],[1, 0]])
        elif rep == "quotient_V":
            ipt_capsule_dim = 6
            data = [1] * 6
            rep_gen = [0, 1, 4, 6, 7, 10]
            row1 = [1,0,10,7,6,4]
            row4 = [4,7,0,10,1,6]
            col = list(range(ipt_capsule_dim))
            rho1 = coo_matrix((data,(row1,col)),shape=(24,ipt_capsule_dim)).toarray()[rep_gen]
            rho4 = coo_matrix((data,(row4,col)),shape=(24,ipt_capsule_dim)).toarray()[rep_gen]
        else:
            raise AssertionError("Wrong group representations！")        
        return g1, g4, rho1, rho4, rho1.shape[0]

    elif group == "A5":
        phi = (1+sqrt(5))/2
        g_a91 = np.array([[(1-phi)/2, phi/2, -1/2],[-phi/2, -1/2, (1-phi)/2],[-1/2, (phi-1)/2, phi/2]])
        g_c01 = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]])
        if rep == "trivial":
            rho_a91 = np.array([1])
            rho_c01 = np.array([1])
        elif rep == "regular":
            e=0;c42=1;c43=2;c41=3;a32=4;a62=5;a92=6;a81=7;a82=8;a91=9;a61=10;a31=11;
            c01=12;a72=13;b02=14;b22=15;a11=16;c32=17;b54=18;b33=19;b14=20;b43=21;c22=22;a21=23;
            c02=24;b44=25;b51=26;a52=27;a22=28;b31=29;b24=30;c12=31;c33=32;b03=33;b12=34;a01=35;
            c03=36;b34=37;a42=38;b11=39;a02=40;b52=41;c23=42;b23=43;b04=44;c13=45;b41=46;a12=47;
            c11=48;a71=49;b53=50;b13=51;c21=52;a51=53;b32=54;b01=55;b42=56;b21=57;a41=58;c31=59;
            ipt_capsule_dim = 60
            data = [1] * ipt_capsule_dim
            row_a91 = [a91,a61,a82,a31,c42,c43,e,c41,a62,a92,a32,a81,b51,a52,c02,b44,b31,a22,c12,
                       b24,a01,b12,b03,c33,b21,a41,b42,c31,a71,b53,c11,b13,a51,b32,c21,b01,c23,a02,
                       b52,b23,b41,b04,c13,a12,a42,c03,b34,b11,b33,c32,a11,b54,b43,a21,c22,b14,c01,b02,b22,a72]
            row_c01 = [c01,a72,b02,b22,a11,c32,b54,b33,b14,b43,c22,a21,e,c42,c43,c41,a32,a62,a92,
                       a81,a82,a91,a61,a31,c03,b34,a42,b11,a02,b52,c23,b23,b04,c13,b41,a12,c02,b44,
                       b51,a52,a22,b31,b24,c12,c33,b03,b12,a01,a71,c11,b13,b53,b01,b32,a51,c21,a41,c31,b42,b21]
            col = list(range(ipt_capsule_dim))
            rho_a91 = coo_matrix((data,(row_a91,col)),shape=(ipt_capsule_dim,ipt_capsule_dim)).toarray()
            rho_c01 = coo_matrix((data,(row_c01,col)),shape=(ipt_capsule_dim,ipt_capsule_dim)).toarray()
        else:
            raise AssertionError("Wrong group representations")
        return g_a91, g_c01, rho_a91, rho_c01, rho_a91.shape[0]

    elif group == "SO2":
        g = np.array([[cos(1), -sin(1), 0],[sin(1), cos(1), 0],[0, 0, 1]])
        if rep == "trivial":
            rho = np.array([1])
        elif rep[:-1] == "irrep":
            k = int(rep[-1])
            rho = np.array([[cos(k), -sin(k), 0],[sin(k), cos(k), 0],[0, 0, 1]])
        else:
            raise AssertionError("Wrong group representations")
        return g, rho, rho.shape[0]

    elif group == "SO3":
        g1 = np.array([[cos(1), 0, sin(1)],[0, 1, 0],[-sin(1), 0, cos(1)]]) 
        g2 = np.array([[1, 0, 0],[0, cos(1), -sin(1)],[0, sin(1), cos(1)]]) 
        if rep == "trivial":
            rho1 = np.array([1])
            rho2 = np.array([1])            
        elif rep[:-1] == "irrep":
            k = int(rep[-1])
            rho1 = wigner_D_matrix(k, 1, 0, 0)
            rho2 = wigner_D_matrix(k, 0, 1, 0)
        else:
            raise AssertionError("Wrong group representations!")
        return g1, g2, rho1, rho2, rho1.shape[0]

    else:
        raise AssertionError("Wrong group!")