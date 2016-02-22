# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 15:18:01 2015

@author: Toshiba
"""
import numpy as np
import matplotlib.pyplot as plt
from Grid_definition import*
from Geometry_definition import*
    
def delta(r, dr):
    res = np.zeros(r.shape)
    mask1 = (r >= 0.5*dr) & (r <= 1.5*dr)
    V1 = (5-3*np.abs(r*mask1)/dr \
        - np.sqrt(-3*(1-np.abs(r*mask1)/dr)**2+1))/(6*dr)
    res[mask1] = V1[mask1]    
    mask2 = np.logical_not(mask1) & (r <= 0.5*dr)
    V2 = (1+np.sqrt(-3*(r*mask2/dr)**2+1))/(3*dr)
    res[mask2] = V2[mask2]
    return res
  
def matrix_E(X,Y,nx,ny,xhi,eta,nk):
  
  dx = np.reshape(np.tile(np.tile(np.concatenate((np.array([np.diff(X)[0,0]]),\
       (np.diff(X)[0,:-1]+np.diff(X)[0,1:])/2.0,np.array([np.diff(X)[0,-1]]))),ny),nk),(nk,-1))
       
  dy = -np.reshape(np.tile(np.concatenate((np.diff(Y,axis = 0)[0,:],  \
     ((np.diff(Y,axis = 0)[:-1,0]+np.diff(Y,axis = 0)[1:,0])/2.0).repeat(nx),\
     np.diff(Y,axis = 0)[-1,:])),nk),(nk,-1))
  X = np.ravel(X)
  Y = np.ravel(Y)
  rx =  abs(X[np.newaxis,:] - xhi[:,np.newaxis])
  ry =  abs(Y[np.newaxis,:] - eta[:,np.newaxis])
  deltx = delta(rx,dx)
  delty = delta(ry,dy)
  E = deltx*delty*dx*dy
  return E
  
if __name__=="__main__":
 
 nx = 65; ny = 65
 nk = 60
 nxu,nyu = nx,ny-1
 nxv,nyv = nx-1,ny
 Lx = Ly = 1.0
 
 X,Y,Xu,Yu,Xv,Yv,Xc,Yc,xv,yu = Staggered_grid_ref1(Lx,Ly,nx,ny,0.3,1)

 
 xhi,eta,alpha = Cylinder(nk,0.3,0.5,0.5)

 
 Eu = matrix_E(Xu,Yu,nxu,nyu,xhi,eta,nk)
 Ev = matrix_E(Xv,Yv,nxv,nyv,xhi,eta,nk)


#Check regularization operator
 uB = np.ones((nk))
 vB = np.zeros((nk))
 u = np.zeros((nyu,nxu))
 v = np.zeros((nyv,nxv))
 u = np.reshape(np.dot(Eu.transpose(),uB),u.shape)
 v = np.reshape(np.dot(Ev.transpose(),vB),v.shape)

 uC=0.5*(u[:,1:] + u[:,:-1])
 vC=0.5*(v[1:,:] + v[:-1,:])


 plt.figure(figsize=(7,7))
 plt.quiver(xv, yu, uC, vC,scale=1,units='xy')
 plt.plot(xhi,eta)
 plt.draw()

 
#Check interpolation operator
 uB = np.zeros((nk))
 vB = np.zeros((nk))
 u = np.ones((nyu,nxu))
 v = np.zeros((nyv,nxv))
 uB = np.dot(Eu,np.ravel(u))
 vB = np.dot(Ev,np.ravel(v))
 
 plt.figure(figsize=(7,7))
 plt.quiver(xhi, eta, uB, vB)
 plt.plot(xhi,eta)
 plt.draw()

 plt.figure()
 plt.spy(uC)

 
 



