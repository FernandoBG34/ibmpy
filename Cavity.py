# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:56:42 2015

@author: Toshiba
"""

import matplotlib.pyplot as plt
import numpy as np
from time import time
from scipy.sparse.linalg import factorized

from Convective_terms import*
from Solvers import*
from Grid_definition import*
from Matrices_definition import*

#Parameters 
Lx = Ly = 1.0
nx = 65; ny = 65
CFL = 0.5
Re = 100.0
Mesh = 1 #0 Uniforme 1 Refinada

#Grid
if Mesh == 0:
 X,Y,Xu,Yu,Xv,Yv,Xc,Yc,xv,yu = Staggered_grid_linspace(Lx,Ly,nx,ny)
else:
 X,Y,Xu,Yu,Xv,Yv,Xc,Yc,xv,yu = Staggered_grid_ref1(Lx,Ly,nx,ny,0.3,0)
 
#dt
dt1 = CFL*min(np.diff(Xu)[0])
dt2 = -CFL*min(np.diff(Yu,axis =0)[0,:])
dt = min(dt1,dt2)
 
#Initial velocities 
u = np.zeros((ny-1,nx)) 
v = np.zeros((ny,nx-1))

#Boundary conditions
UN = np.ones((nx-2));  VN = np.zeros((nx-1))
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
BN = dt*M_inv + dt**2/2.0*(M_inv.dot(L)).dot(M_inv) + dt**3/(2.0**2)*((M_inv.dot(L)).dot(M_inv.dot(L))).dot(M_inv)

GtBnG = G.T.dot(BN).dot(G)
iGtBnG = factorized(GtBnG)

#First step
unm1 = u
vnm1 = v

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
   Phi = solver_direct(iGtBnG, -D.dot(q) + bc2)  
   
   #Corrector step   
   qn = q - BN.dot(G.dot(Phi))
   Veln = R_inv.dot(qn)
   
   Un = np.zeros_like(u); Vn = np.zeros_like(v)

   Un[:,1:-1] = Veln[:(ny-1)*(nx-2)].reshape(ny-1,nx-2)
   Vn[1:-1,:] = Veln[-(nx-1)*(ny-2):].reshape(ny-2,nx-1)
   
   #Residue
   if k >= 1:
     Res = L2_norm(u,unm1,dt)
   
   #Actualization
   unm1,vnm1 = u,v
   u,v = Un,Vn
   
   k = k+1
   print "Iter=", k, "### Res=", Res
   
time9 = time()
print "Tiempo de ejecución", time9-time0

###PLOTS###

#np.savez('Cavity_Re={}_n={}_mesh={}'.format(Re,nx,Mesh), Un[:,nx/2],Vn[ny/2,:],yu,xv)
npzfile = np.load("Data_ghia.npz")

x_ghia = npzfile['arr_0']
y_ghia = npzfile['arr_1']
#Re100
u_ghia_100 = npzfile['arr_2']
v_ghia_100 = npzfile['arr_3']
####Re1000
u_ghia_1000 = npzfile['arr_4']
v_ghia_1000 = npzfile['arr_5']
###Re3200
u_ghia_3200 = npzfile['arr_6']
v_ghia_3200 = npzfile['arr_7']


plt.figure(1)
plt.plot(Un[:,nx/2], yu)
plt.xlabel('u')
plt.ylabel('y')

plt.figure(2)
plt.plot(xv,Vn[ny/2,:])
plt.xlabel('x')
plt.ylabel('v')

plt.figure(1)
plt.plot(u_ghia_100,y_ghia,'o')
plt.figure(2)
plt.plot(x_ghia,v_ghia_100,'o')

uC=0.5*(u[:,1:] + u[:,:-1])
vC=0.5*(v[1:,:] + v[:-1,:])
plt.figure(figsize=(7,7))
plt.streamplot(xv, yu, uC, vC, \
                density=4, linewidth=0.5, color='k', arrowsize=0.1)
plt.xlabel('X')
plt.ylabel('Y')    
        