import numpy as np
from math import *

def discretized_PDO(mode='Gaussian',hx=1,hy=1,hz=1,r=2,sigma=.9):
    if mode == 'FD':
        hx = 1
        hy = 1
        hz = 1
        
        p = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        p_x = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [-1/2, 0, 1/2], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])/hx

        p_y = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[0, -1/2, 0], [0, 0, 0], [0, 1/2, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])/hy

        p_z = np.array([[[0, 0, 0], [0, -1/2, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 1/2, 0], [0, 0, 0]]])/hz

        p_xx = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [1, -2, 1], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])/hx/hx

        p_xy = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[-1/4, 0, 1/4], [0, 0, 0], [1/4, 0, -1/4]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])/hx/hy

        p_xz = np.array([[[0, 0, 0], [-1/4, 0, 1/4], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [1/4, 0, -1/4], [0, 0, 0]]])/hx/hz

        p_yy = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[0, 1, 0], [0, -2, 0], [0, 1, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])/hy/hy

        p_yz = np.array([[[0, -1/4, 0], [0, 0, 0], [0, 1/4, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[0, 1/4, 0], [0, 0, 0], [0, -1/4, 0]]])/hy/hz

        p_zz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
              [[0, 0, 0], [0, -2, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])/hz/hz
        
    elif mode == 'Gaussian':
        def Gaussian(x,y,z,sigma=.9):
            return 1/((2*pi)**1.5*sigma**3)*exp(-(x**2+y**2+z**2)/2/sigma**2)
        
        def Gaussian_x(x,y,z,sigma=.9):
            return x/(-(2*pi)**1.5*sigma**5)*exp(-(x**2+y**2+z**2)/2/sigma**2)
        
        def Gaussian_y(x,y,z,sigma=.9):
            return y/(-(2*pi)**1.5*sigma**5)*exp(-(x**2+y**2+z**2)/2/sigma**2)

        def Gaussian_z(x,y,z,sigma=.9):
            return z/(-(2*pi)**1.5*sigma**5)*exp(-(x**2+y**2+z**2)/2/sigma**2)        
        
        def Gaussian_xx(x,y,z,sigma=.9):
            return (1-x**2/sigma**2)/(-(2*pi)**1.5*sigma**5)*exp(-(x**2+y**2+z**2)/2/sigma**2)

        def Gaussian_yy(x,y,z,sigma=.9):
            return (1-y**2/sigma**2)/(-(2*pi)**1.5*sigma**5)*exp(-(x**2+y**2+z**2)/2/sigma**2)
        
        def Gaussian_zz(x,y,z,sigma=.9):
            return (1-z**2/sigma**2)/(-(2*pi)**1.5*sigma**5)*exp(-(x**2+y**2+z**2)/2/sigma**2)

        def Gaussian_xy(x,y,z,sigma=.9):
            return (x*y)/((2*pi)**1.5*sigma**7)*exp(-(x**2+y**2+z**2)/2/sigma**2)
        
        def Gaussian_xz(x,y,z,sigma=.9):
            return (x*z)/((2*pi)**1.5*sigma**7)*exp(-(x**2+y**2+z**2)/2/sigma**2)        

        def Gaussian_yz(x,y,z,sigma=.9):
            return (y*z)/((2*pi)**1.5*sigma**7)*exp(-(x**2+y**2+z**2)/2/sigma**2)
        
        r = 2  # radius
        n = 2 * r + 1
#         sigma = .5
        p = np.zeros((n,n,n))
        p_x = np.zeros((n,n,n))
        p_y = np.zeros((n,n,n))
        p_z = np.zeros((n,n,n))
        p_xx = np.zeros((n,n,n))
        p_xy = np.zeros((n,n,n))
        p_xz = np.zeros((n,n,n))
        p_yy = np.zeros((n,n,n))
        p_yz = np.zeros((n,n,n))
        p_zz = np.zeros((n,n,n))

        for i in range(n):
            x = i - r
            for j in range(n):
                y = j - r
                for k in range(n):
                    z = k - r
                    p[i,j,k] = Gaussian(x,y,z,sigma)
                    p_x[i,j,k] = Gaussian_x(x,y,z,sigma)
                    p_y[i,j,k] = Gaussian_y(x,y,z,sigma)
                    p_z[i,j,k] = Gaussian_z(x,y,z,sigma)
                    p_xx[i,j,k] = Gaussian_xx(x,y,z,sigma)
                    p_xy[i,j,k] = Gaussian_xy(x,y,z,sigma)
                    p_xz[i,j,k] = Gaussian_xz(x,y,z,sigma)
                    p_yy[i,j,k] = Gaussian_yy(x,y,z,sigma)
                    p_yz[i,j,k] = Gaussian_yz(x,y,z,sigma)
                    p_zz[i,j,k] = Gaussian_zz(x,y,z,sigma)
                
    total_partial = np.stack((p,p_x,p_y,p_z,p_xx,p_xy,p_xz,p_yy,p_yz,p_zz))
    return total_partial