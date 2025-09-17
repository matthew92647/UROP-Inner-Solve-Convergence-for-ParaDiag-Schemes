from math import pi
from circulant import (
    sparse_circulant,
    gradient_stencil,
    BlockCirculantLinearOperatorExact,
    BlockCirculantLinearOperatorInexact
)
import numpy as np
from scipy.sparse import linalg as spla
from scipy import sparse


nt = 128
nx = 128

lx = 2*pi
dx = lx/nx

theta = 1.0

# velocity, CFL, and reynolds number
u = 1
re = 500
cfl = 0.8

# viscosity and timestep
nu = lx*u/re
dt = cfl*dx/u

# advective and diffusive Courant numbers

cfl_u = cfl
cfl_v = nu*dt/dx**2

print(f"{nu = }, {dt = }, {cfl_v = }, {cfl_u = }")  # noqa E251

# Spatial domain
mesh = np.linspace(start=-lx/2, stop=lx/2, num=nx, endpoint=False)

# Mass matrix
M = sparse_circulant([1], nx)

# Advection matrix
D = sparse_circulant(gradient_stencil(1, order=2), nx)

# Diffusion matrix
L = sparse_circulant(gradient_stencil(2, order=2), nx)

# Spatial terms
K_advec_diff = (u/dx)*D - (nu/dx**2)*L
K_advec = (u/dx)*D 
K_diff = - (nu/dx**2)*L

I = sparse.identity(K_advec_diff.shape[0], format='csc')

A_advec_diff = I + dt*K_advec_diff
A_advec = I + dt*K_advec
A_diff = I + dt*K_diff

A_advec_diff_lu = spla.splu(A_advec_diff.tocsc())
A_advec_lu = spla.splu(A_advec.tocsc())
A_diff_lu = spla.splu(A_diff.tocsc())

A_advec_diff_inv = A_advec_diff_lu.solve(np.eye(K_advec_diff.shape[0]))
A_advec_inv = A_advec_lu.solve(np.eye(K_advec.shape[0]))
A_diff_inv = A_diff_lu.solve(np.eye(K_diff.shape[0]))

# Compute norms
norm_advec_diff_inf = np.linalg.norm(A_advec_diff_inv, np.inf)
norm_advec_diff_2 = np.linalg.norm(A_advec_diff_inv, 2)

norm_advec_inf = np.linalg.norm(A_advec_inv, np.inf)
norm_advec_2 = np.linalg.norm(A_advec_inv, 2)

norm_diff_inf = np.linalg.norm(A_diff_inv, np.inf)
norm_diff_2 = np.linalg.norm(A_diff_inv, 2)

print("Advection Diffusion: ")
print("|| (I + dt K)^-1 ||_inf =", norm_advec_diff_inf)
print("|| (I + dt K)^-1 ||_2   =", norm_advec_diff_2)

print("\n")

print("Advection:")
print("|| (I + dt K)^-1 ||_inf =", norm_advec_inf)
print("|| (I + dt K)^-1 ||_2   =", norm_advec_2)

print("\n")

print("Diffusion:")
print("|| (I + dt K)^-1 ||_inf =", norm_diff_inf)
print("|| (I + dt K)^-1 ||_2   =", norm_diff_2)
