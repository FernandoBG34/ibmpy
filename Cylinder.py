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
dxM = dyM = 0.8
dxm = dym = 0.04
nx = ny = 76
CFL = 0.4
Re = 200.0
R1 = 0.5
Nk = 78

#Grid
X,Y,Xu,Yu,Xv,Yv,Xc,Yc,xv,yu = Staggered_grid_ref2(nx,ny,dxm,dym,dxM,dyM,0.12)
nx = nx*2-1
ny = ny*2-1

#dt
dt1 = CFL*min(np.diff(X)[0])
dt2 = -CFL*min(np.diff(Y,axis =0)[0,:])
dt = min(dt1,dt2)

#Geometry
xhi, eta, alpha = Cylinder(Nk,R1,np.amax(X)/2.0,np.amax(Y)/2.0)

#Initial velocities 
u = fx = np.zeros((ny-1,nx))
v = fy = np.zeros((ny,nx-1))
uB = np.zeros((Nk))
vB = np.zeros((Nk))

#Boundary conditions
Uinf = 1.0
UN = np.ones((nx-2)); VN = np.zeros((nx-1))
US = np.ones((nx-2)); VS = np.zeros((nx-1))
UW = np.ones((ny-1)); VW = np.zeros((ny-2))
UE = np.ones((ny-1)); VE = np.zeros((ny-2))

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
Lift = np.zeros((500000))
Drag = np.zeros((500000))
#for n in range(500):
while (Res > 1e-5):   
    
   #Right-Hand-Side 
   Nu = Adv_u(u,v,Xc,Y,UN,US);            Nv = Adv_v(u,v,X,Yc,VE,VW)
   Nunm1 = Adv_u(unm1,vnm1,Xc,Y,UN,US);   Nvnm1 = Adv_v(unm1,vnm1,X,Yc,VE,VW)
   
   rn_hat = np.concatenate((Ru.dot(u[:,1:-1].ravel()) + np.ravel((3*Nu-Nunm1)/2.0), \
                            Rv.dot(v[1:-1,:].ravel()) + np.ravel((3*Nv-Nvnm1)/2.0)))
                        
   rn = M_hat.dot(rn_hat)
    
   #Outflow boundary condition
   bc1_hat = np.concatenate(((UN*BUN + US*BUS + UE*BUE + UW*BUW)/Re ,(VN*BVN + VS*BVS + VE*BVE + VW*BVW)/Re))
   bc1 = M_hat.dot(bc1_hat)

   #Intermediate velocity
   q = solver_iterative(A,rn+bc1)
   
   #Poisson
   r2 = np.concatenate((bc2,np.concatenate((uB,vB))))
   Lambda = solver_direct(iQtBnQ, Q.T.dot(q) - r2) 
      
   #Corrector step   
   qn = q - BN.dot(Q).dot(Lambda)  
   Veln = R_inv.dot(qn)
   
   Un = np.ones_like(u); Vn = np.zeros_like(v)

   Un[:,1:-1] = Veln[:(ny-1)*(nx-2)].reshape(ny-1,nx-2)
   Vn[1:-1,:] = Veln[-(nx-1)*(ny-2):].reshape(ny-2,nx-1)
   
   #Forces
   Fx_hat = np.concatenate((Lambda[-2*Nk:-Nk],np.zeros((Nk))))
   Fy_hat = np.concatenate((np.zeros((Nk)),Lambda[-Nk:]))
   Drag[k] = 2*sum(M_hat_inv.dot(E.T).dot(Fx_hat))*dxm*dym
   Lift[k] = 2*sum(M_hat_inv.dot(E.T).dot(Fy_hat))*dxm*dym

   #Residue
   if k >= 1:
     Res = L2_norm(u,unm1,dt)
   
   #Actualization
   unm1,vnm1 = u.copy(),v.copy()
   u,v = Un.copy(),Vn.copy()
   UE = u[:,-1] - Uinf*(u[:,-1]-u[:,-2])*dt/np.diff(Xu)[0,-1]

   k = k+1
   print "Iter=", k, "### Res=", Res, "Cd = ", Drag[k-1], "Cl = ", Lift[k-1]
   
time9 = time()
print "Tiempo de ejecución", time9-time0

plt.figure(figsize = (9,7))
plt.pcolormesh(Xu,Yu,Un)
plt.colorbar()
plt.plot(xhi,eta,'k-')
plt.xlim(0,np.amax(X))
plt.ylim(0,np.amax(Y))

plt.figure()
plt.plot(Xu[ny/2,:],Un[ny/2,:])
plt.plot(np.array([0,np.amax(X)]),np.array([0,0]))

plt.figure()
plt.plot(np.linspace(0,k*dt,k),Lift[:k],label = 'Lift')
plt.plot(np.linspace(0,k*dt,k),Drag[:k],label = 'Drag')
