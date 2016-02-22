# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 12:03:09 2016

@author: Toshiba
"""

import numpy as np
from scipy import sparse as sp
from scipy.sparse import vstack, hstack
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm

from Grid_definition import*

 
def dcentered(x,xv):
  nx = len(x)
  dx = np.diff(xv)
  dxbcw = xv[0]-x[0]
  dxbce = x[-1]-xv[-1]
  
  xmid = np.concatenate((np.array([-2/(dxbcw*dx[0])]),-2/(dx[:-1]*dx[1:]),np.array([-2/(dxbce*dx[-1])])))
  xleft = np.concatenate((2/(dx[:-1]*(dx[:-1]+dx[1:])),np.array([2/(dx[-1]*(dx[-1]+dxbce))])))
  xright = np.concatenate((np.array([2/(dx[0]*(dx[0]+dxbcw))]),2/(dx[1:]*(dx[:-1]+dx[1:]))))
     
  xdiagonals = [xleft,xmid,xright]
  Dcen = sp.diags(xdiagonals, [-1,0,1])
    
  bw = np.zeros((nx-1)); bw[0] =  2/(dxbcw*(dx[0]+dxbcw))
  be = np.zeros((nx-1)); be[-1] = 2/(dxbce*(dx[-1]+dxbce))
    
  return Dcen, bw, be
    
def dnodos(x):
    
  nx = len(x)-2
  dx = np.diff(x)
    
  mid = -2/(dx[:-1]*dx[1:])
  left = 2/(dx[1:-1]*(dx[1:-1]+dx[2:]))
  right = 2/(dx[1:-1]*(dx[1:-1]+dx[:-2]))
    
  diagonals = [left,mid,right]
  Dnod = sp.diags(diagonals, [-1,0,1])
    
  bn = np.zeros((nx)); bn[0] =  2/(dx[0]*(dx[0]+dx[1]))
  bs = np.zeros((nx)); bs[-1] = 2/(dx[-1]*(dx[-1]+dx[-2]))
    
  return Dnod, bn, bs
    
def U_matrices(Y,Yu,Xu,nx,ny,dt,Re):
     
  Dnod,bw,be = dnodos(Xu[0,:])  
  Dcen,bn,bs = dcentered(Y[:,0],Yu[:,0]) 
 
  LU = sp.kron(Dcen,sp.eye(nx-2)) + sp.kron(sp.eye(ny-1),Dnod)
  BUW = sp.kron(sp.eye(ny-1),bw); BUE = sp.kron(sp.eye(ny-1),be);
  BUN = sp.kron(bn,sp.eye(nx-2)); BUS = sp.kron(bs,sp.eye(nx-2))
  Au = sp.eye(LU.shape[0])/dt - 1/(2.0*Re)*LU
  Ru = sp.eye(LU.shape[0])/dt + 1/(2.0*Re)*LU
  
  return LU,BUN,BUS,BUW,BUE,Au,Ru
  
def V_matrices(X,Xv,Yv,nx,ny,dt,Re):
     
  Dnod,bn,bs = dnodos(Yv[:,0])  
  Dcen,bw,be = dcentered(X[0,:],Xv[0,:]) 
 
  LV = sp.kron(sp.eye(ny-2),Dcen) + sp.kron(Dnod,sp.eye(nx-1))
  BVW = sp.kron(sp.eye(ny-2),bw); BVE = sp.kron(sp.eye(ny-2),be);
  BVN = sp.kron(bn,sp.eye(nx-1)); BVS = sp.kron(bs,sp.eye(nx-1))
  Av = sp.eye(LV.shape[0])/dt - 1/(2.0*Re)*LV
  Rv = sp.eye(LV.shape[0])/dt + 1/(2.0*Re)*LV
  
  return LV,BVN,BVS,BVW,BVE,Av,Rv
  
def bc2 (X,Y,nx,ny):
    
  dx = sp.diags(np.diff(X[0,:]),0)
  dy = sp.diags(-np.diff(Y[:,0]),0)
   
  bw = np.zeros((nx-1)); bw[0] =  -1
  be = np.zeros((nx-1)); be[-1] = 1
  bn = np.zeros((ny-1)); bn[0] =  1
  bs = np.zeros((ny-1)); bs[-1] = -1
   
  B2UW = sp.kron(dy,bw)
  B2UE = sp.kron(dy,be)
  B2VN = sp.kron(bn,dx)
  B2VS = sp.kron(bs,dx)
    
  return B2UW, B2UE, B2VN, B2VS
    

def GMR_matrices(nx,ny,X,Y,Xc,Yc):

  nxp = nx-1; nyp = ny-1
 
  dyj = sp.diags(-np.diff(Y[:,0]),0)
  dxi = sp.diags(np.diff(X[0,:]),0)
  Dyj = sp.kron(dyj,sp.eye(nx-2))
  Dxi = sp.kron(sp.eye(ny-2),dxi)
  R = sp.block_diag((Dyj,Dxi))
               
  idyj = sp.diags(-1.0/np.diff(Y[:,0]),0)
  idxi = sp.diags(1.0/np.diff(X[0,:]),0)
  iDyj = sp.kron(idyj,sp.eye(nx-2))
  iDxi = sp.kron(sp.eye(ny-2),idxi)
  R_inv = sp.block_diag((iDyj,iDxi))
   
   
  iDyj = sp.kron(idyj,sp.eye(nx-1))
  iDxi = sp.kron(sp.eye(ny-1),idxi)
   
  dyc = sp.diags(-np.diff(Yc[:,0]),0)
  dxc = sp.diags(np.diff(Xc[0,:]),0)
  Iyc = np.ones(nyp)
  Ixc = np.ones(nxp)
  Dyc = sp.kron(dyc,sp.diags(Ixc, 0, (nxp, nxp)))
  Dxc = sp.kron(sp.diags(Iyc, 0, (nyp, nyp)),dxc)
  M_hat = sp.block_diag((Dxc,Dyc))

  idyc = sp.diags(-1.0/np.diff(Yc[:,0]),0)
  idxc = sp.diags(1.0/np.diff(Xc[0,:]),0)
  iDyc = sp.kron(idyc,sp.eye(nxp))
  iDxc = sp.kron(sp.eye(nyp),idxc)
  M_hat_inv = sp.block_diag((iDxc,iDyc))

  Aux_x = sp.diags([-1,1],[0,1],(nxp-1,nxp))
  Aux_y = sp.diags([1,-1],[0,1],(nyp-1,nyp))
  Gx = sp.kron(sp.eye(nyp),Aux_x)
  Gy = sp.kron(Aux_y, sp.eye(nxp))
  G  = vstack((Gx,Gy))
    
  G_hat = M_hat_inv.dot(G)
 
  return G_hat, R, R_inv, M_hat, M_hat_inv
  

    

