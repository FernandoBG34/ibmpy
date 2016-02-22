# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:50:39 2015

@author: Toshiba
"""

import numpy as np

def Cylinder(nk,R,x_dist,y_dist):
 alpha = 2*np.pi*np.arange(0,nk)/nk
 xhi = x_dist + R*np.cos(alpha)
 eta = y_dist + R*np.sin(alpha)
 return xhi, eta, alpha
 
 
def Concentric_cylinders(nk,R,R2,x_dist,y_dist):
 alpha = 2*np.pi*np.arange(0,nk)/nk
 xhi = x_dist + R*np.cos(alpha)
 eta = y_dist + R*np.sin(alpha)
 
 nk2 = int(R2*nk/R) 
 alpha2 = 2*np.pi*np.arange(0,nk2)/nk2
 xhi2 = x_dist + R2*np.cos(alpha2)
 eta2 = y_dist + R2*np.sin(alpha2)
 
 XHI, ETA = np.concatenate((xhi,xhi2)), np.concatenate((eta,eta2))
 return XHI, ETA, alpha, alpha2, nk2