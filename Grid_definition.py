# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:44:13 2015

@author: Toshiba
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

def stretching(n, dn0, dn1, ns, ws, we, maxs):
    ne = ns + np.log(dn1/dn0)/np.log(1+maxs)

    s=np.array([maxs*0.25*(1+erf(6*(x-ns)/(ws)))*(1-erf(6*(x-ne)/we)) for x in range(n)])

    f_=np.empty(s.shape); f_[0] = dn0
    for k in range(1,len(f_)):
      f_[k] = f_[k-1]*(1+s[k])
    f=np.empty(s.shape);  f[0] = 0.0
    for k in range(1,len(f)):
      f[k] = f[k-1] + f_[k]

    return f

def Staggered_grid_linspace(Lx,Ly,nx,ny):
   
  x = np.linspace(0,Lx,nx)
  y = np.linspace(Ly,0,ny)
  ##################################
  xv = x[:-1] + np.diff(x)/2.0
  yv = y
  xu = x
  yu = y[:-1] + np.diff(y)/2.0
  xc = xv
  yc = yu
  X,Y = np.meshgrid(x,y)
  Xu,Yu = np.meshgrid(xu,yu)
  Xv,Yv = np.meshgrid(xv,yv)
  Xc,Yc = np.meshgrid(xc,yc)
   
  return X,Y,Xu,Yu,Xv,Yv,Xc,Yc,xv,yu
  

def Staggered_grid_ref1(Lx,Ly,nx,ny,m,z): 
  #z = 0 Paredes, 1 Centro.0.1<m<0.7. Cuanto mayor menos brusco.   
  r = np.linspace(m,m+0.5,(nx/2)+1)
  f = 1/np.sin(r)
  g = Lx/2.0*f/(f[0]-f[-1])
  g = g-g[-1]
  x = np.concatenate((g[::-1],(Lx-g)[1:]))
  if z == 1:
   l = -(g[::-1]-g[::-1][-1])
   x = np.concatenate((l[::-1],(Lx-l)[1:]))
  #########
  r2 = np.linspace(m,m+0.5,(ny/2)+1)
  f2 = 1/np.sin(r2)
  g2 = Ly/2.0*f2/(f2[0]-f2[-1])
  g2 = g2-g2[-1]
  x2 = np.concatenate((g2[::-1],(Ly-g2)[1:]))
  if z == 1:
   l = -(g[::-1]-g[::-1][-1])
   x2 = np.concatenate((l[::-1],(Ly-l)[1:]))
  y = x2[::-1] 
  #############################################  
  xv = x[:-1] + np.diff(x)/2.0
  yv = y
  xu = x
  yu = y[:-1] + np.diff(y)/2.0
  xc = xv
  yc = yu
  X,Y = np.meshgrid(x,y)
  Xu,Yu = np.meshgrid(xu,yu)
  Xv,Yv = np.meshgrid(xv,yv)
  Xc,Yc = np.meshgrid(xc,yc)
  return X,Y,Xu,Yu,Xv,Yv,Xc,Yc,xv,yu
  
 
def Staggered_grid_ref2(nx,ny,dxm,dym,dxM,dyM,maxs):
  s1 = stretching (nx,dxm,dxM,int(0.7/dxm), 16,16,maxs)
  x = np.concatenate([-s1[::-1], s1[1:]])
  x = x - np.amin(x)
  s2 = stretching (ny,dym,dyM,int(0.7/dym), 16,16,maxs)
  y = np.concatenate([-s2[::-1], s2[1:]])
  y = y - np.amin(y)
  y = y[::-1]
  ###################################  
  xv = x[:-1] + np.diff(x)/2.0
  yv = y
  xu = x
  yu = y[:-1] + np.diff(y)/2.0
  xc = xv
  yc = yu
  X,Y = np.meshgrid(x,y)
  Xu,Yu = np.meshgrid(xu,yu)
  Xv,Yv = np.meshgrid(xv,yv)
  Xc,Yc = np.meshgrid(xc,yc)
  return X,Y,Xu,Yu,Xv,Yv,Xc,Yc,xv,yu
  
def Staggered_grid_ref3(nx,ny,nx2,dxm,dym,dxM,dxMM,dyM,maxs):
  s1 = stretching (nx,dxm,dxM,int(0.7/dxm), 16,16,maxs)
  s11 = stretching (nx2,dxm,dxMM,int(0.7/dxm), 16,16,maxs)
  x = np.concatenate([-s1[::-1], s11[1:]])
  x = x - np.amin(x)
  s2 = stretching (ny,dym,dyM,int(0.7/dym), 16,16,maxs)
  y = np.concatenate([-s2[::-1], s2[1:]])
  y = y - np.amin(y)
  y = y[::-1]
  ###################################  
  xv = x[:-1] + np.diff(x)/2.0
  yv = y
  xu = x
  yu = y[:-1] + np.diff(y)/2.0
  xc = xv
  yc = yu
  X,Y = np.meshgrid(x,y)
  Xu,Yu = np.meshgrid(xu,yu)
  Xv,Yv = np.meshgrid(xv,yv)
  Xc,Yc = np.meshgrid(xc,yc)
  return X,Y,Xu,Yu,Xv,Yv,Xc,Yc,xv,yu, np.amax(s1)

  

if __name__ == "__main__":
 
 dxM = dyM = 0.33
 dxm = dym = 0.02
 nx = 151
 ny = 151
 X,Y,Xu,Yu,Xv,Yv,Xc,Yc,xv,yu = Staggered_grid_ref2(nx,ny,dxm,dym,dxM,dyM,0.12)

# plt.figure(figsize = (7,7))
# plt.plot(X)
# plt.figure(figsize = (7,7))
# plt.plot(Y.transpose())
 
 print np.amax(X)
 print np.amax(np.diff(X)[0,:])/np.amin(np.diff(X)[0,:])
 

 

