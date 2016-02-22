# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:56:42 2015

@author: Toshiba
"""

import matplotlib.pyplot as plt
import numpy as np
from time import time
from scipy.sparse.linalg import factorized
from scipy.linalg import norm

from Convective_terms import*
from Solvers import*
from Grid_definition import*
from Matrices_definition import*
from Geometry_definition import*
from E_operator import*

#Parameters 
Lx = Ly = 2.1
nx = 65; ny = 65
#CFL = 0.2
Re = 100.0
R1 = 1.0; R2 = 0.5
nk1 = int(2*np.pi*R1/(Lx/(nx-1)))
Centre = Lx/2.0

#Grid
X,Y,Xu,Yu,Xv,Yv,Xc,Yc,xv,yu = Staggered_grid_linspace(Lx,Ly,nx,ny)

#dt
dt1 = CFL*min(np.diff(X)[0])
dt2 = -CFL*min(np.diff(Y,axis =0)[0,:])
dt = 0.01 #min(dt1,dt2)

#Geometry
xhi, eta, alpha, alpha2, nk2 = Concentric_cylinders(nk1,R1,R2,Centre,Centre)
Nk = nk1+nk2

#Initial velocities 
u = fx = np.zeros((ny-1,nx))
v = fy = np.zeros((ny,nx-1))
uB = np.zeros((Nk)); uB[-nk2:] = np.sin(alpha2)
vB = np.zeros((Nk)); vB[-nk2:] = -np.cos(alpha2)

#Boundary conditions
UN = np.zeros((nx-2)); VN = np.zeros((nx-1))
US = np.zeros((nx-2)); VS = np.zeros((nx-1))
UW = np.zeros((ny-1)); VW = np.zeros((ny-2))
UE = np.zeros((ny-1)); VE = np.zeros((ny-2))

#Matrices
Lu,BUN,BUS,BUW,BUE,Au,Ru = U_matrices(Y,Yu,Xu,nx,ny,dt,Re)
Lv,BVN,BVS,BVW,BVE,Av,Rv = V_matrices(X,Xv,Yv,nx,ny,dt,Re)
L_hat = sp.block_diag((Lu,Lv))
A_hat = sp.block_diag((Au,Av))

bc1_hat = np.concatenate(((UN*BUN + US*BUS + UE*BUE + UW*BUW)/Re ,(VN*BVN + VS*BVS + VE*BVE + VW*BVW)/Re))

B2UW, B2UE, B2VN, B2VS = bc2(X,Y,nx,ny)
bc2 = UE*B2UE + UW*B2UW  + VN*B2VN + VS*B2VS 
 
G_hat, R, R_inv, M_hat, M_hat_inv = GMR_matrices(nx,ny,X,Y,Xc,Yc)

A = M_hat.dot(A_hat).dot(R_inv)
L = M_hat.dot(L_hat).dot(R_inv)/Re
G = M_hat.dot(G_hat)
D = -G.T
bc1 = M_hat.dot(bc1_hat)
M_inv = R.dot(M_hat_inv) 
BN = dt*M_inv + dt**2/2.0*(M_inv.dot(L)).dot(M_inv) + dt**3/(2.0**2)*((M_inv.dot(L))**2).dot(M_inv)

Eu = matrix_E(Xu[:,1:-1],Yu[:,1:-1],nx-2,ny-1,xhi,eta,Nk)
Ev = matrix_E(Xv[1:-1,:],Yv[1:-1,:],nx-1,ny-2,xhi,eta,Nk)
E_hat = sp.block_diag((Eu,Ev))
E = E_hat.dot(R_inv)

Q = hstack((G,E.T))
QtBnQ = Q.T.dot(BN).dot(Q)
iQtBnQ = factorized(QtBnQ)

#First step
unm1 = u.copy()
vnm1 = v.copy()

#Loop
k = 0
Res = 1
time0 = time()
#for n in range(1000):
while (Res > 1e-5):   
    
   #Right-Hand-Side 
   Nu = Adv_u(u,v,Xc,Y,UN,US);            Nv = Adv_v(u,v,X,Yc,VE,VW)
   Nunm1 = Adv_u(unm1,vnm1,Xc,Y,UN,US);   Nvnm1 = Adv_v(unm1,vnm1,X,Yc,VE,VW)
   
   rn_hat = np.concatenate((Ru.dot(u[:,1:-1].ravel()) + np.ravel((3*Nu-Nunm1)/2.0), \
                            Rv.dot(v[1:-1,:].ravel()) + np.ravel((3*Nv-Nvnm1)/2.0)))
                        
   rn = M_hat.dot(rn_hat)
        
   #Intermediate velocity
   q = solver_iterative(A,rn+bc1)
   
   #Poisson
   r2 = np.concatenate((-bc2,np.concatenate((uB,vB))))
   Lambda = solver_direct(iQtBnQ, Q.T.dot(q) - r2) 
      
   #Corrector step   
   qn = q - BN.dot(Q).dot(Lambda)  
   Veln = R_inv.dot(qn)
   
   Un = np.zeros_like(u); Vn = np.zeros_like(v)

   Un[:,1:-1] = Veln[:(ny-1)*(nx-2)].reshape(ny-1,nx-2)
   Vn[1:-1,:] = Veln[-(nx-1)*(ny-2):].reshape(ny-2,nx-1)

   #Residue
   if k >= 1:
     Res = L2_norm(u,unm1,dt)
   
   #Actualization
   unm1,vnm1 = u.copy(),v.copy()
   u,v = Un.copy(),Vn.copy()
   
   k = k+1
   print "Iter=", k, "### Res=", Res
   
time9 = time()
print "Tiempo de ejecución", time9-time0

###PLOTS###

w1 = 0
w2 = 1/R2

x = xv-Lx/2.0
V_analytical = np.zeros_like(x)

for i in range (len(xv)):
 if -R1 < x[i] < -R2 or R2 < x[i] < R1:
   V_analytical[i] = w2*R2*(R1/x[i]-x[i]/R1)/(R2/R1-R1/R2)
 elif -R2 < x[i] < R2:
   V_analytical[i] = -w2*x[i] 
 else:
   V_analytical[i] = 0   
    


from matplotlib import rc

rc('text', usetex=True)
plt.rc('font', family='Computer Modern Roman')
plt.close('all')

plt.figure()
plt.plot(xv,Vn[ny/2,:],label ='Num')
plt.plot(x+Lx/2.0,V_analytical,label ='Exacta')
plt.xlabel('x')
plt.ylabel('v')
plt.xlim(np.amin(Xv),np.amax(Xv))
plt.legend()
plt.savefig("concentricvelocity2.png")



alpha = np.linspace(0,2*np.pi*R1,100)   ;alpha2 = np.linspace(0,2*np.pi,100)              
XHI = Centre + R1*np.cos(alpha);XHI2 = Centre + R2*np.cos(alpha2)
ETA = Centre + R1*np.sin(alpha);ETA2 = Centre + R2*np.sin(alpha2)
plt.figure()                
plt.pcolormesh(Xv,Yv,Vn)
plt.colorbar()
plt.plot(XHI,ETA,'k-')
plt.plot(XHI2,ETA2,'k-')
plt.xlim(np.amin(Xv),np.amax(Xv))
plt.ylim(np.amin(Yv),np.amax(Yv))
plt.xlabel('x')
plt.ylabel('y')






