# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:40:34 2015

@author: Toshiba
"""

import numpy as np
from scipy.sparse import linalg as sla

def solver_direct(A,b):    
  RHS = np.ravel(b)
  solution = A(RHS)
  return solution
  
def solver_iterative(A,b):
  RHS = np.ravel(b)
  solution = sla.cg(A,RHS,tol=1E-10)
  return solution[0]
    
def L2_norm(A,B,dt):
  Res = np.sqrt(np.sum((A-B)**2)/(np.sum(A**2)))/dt
  return Res  
   
