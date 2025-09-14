from math import pi
import numpy as np
from circulant import (
    sparse_circulant,
    gradient_stencil,
    BlockCirculantLinearOperatorExact,
    BlockCirculantLinearOperatorInexact
)
from scipy import sparse
from scipy import linalg
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
from numerical import modified_richardson
from sklearn.linear_model import LinearRegression

nt = 128
nx = 128

lx = 2*pi
dx = lx/nx

theta = 0.5

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

print(f"{nu = }, {dt = }, {cfl_v = }, {cfl_u = }")

# Spatial domain
mesh = np.linspace(start=-lx/2, stop=lx/2, num=nx, endpoint=False)

# Mass matrix                                                                                                                                                                                 
M = sparse_circulant([1], nx)                                                                                                                                                                 
                                                                                                                                                                                              
# Advection matrix                                                                                                                                                                            
D = sparse_circulant(gradient_stencil(1, order=2), nx)                                                                                                                                        
                                                                                                                                                                                              
# Diffusion matrix                                                                                                                                                                            
L = sparse_circulant(gradient_stencil(2, order=2), nx)

# Spatial terms                                                                                                                                                                               
K = (u/dx)*D - (nu/dx**2)*L

# Generate block matrices for different coefficients 


def block_matrix(l1, l2):                                                                                                                                                                     
    mat = l1*M + l2*K                                                                                                                                                                         
    mat.solve = spla.factorized(mat.tocsc())                                                                                                                                                  
    return mat


# Build the full B1 & B2 matrices
b1col = np.zeros(nt)
b1col[0] = 1/dt
b1col[1] = -1/dt

b1row = np.zeros_like(b1col)
b1row[0] = b1col[0]

b2col = np.zeros(nt)
b2col[0] = theta
b2col[1] = 1-theta

b2row = np.zeros_like(b2col)
b2row[0] = b2col[0]

B1 = linalg.toeplitz(b1col, b1row)                                                                                                                                                            
B2 = linalg.toeplitz(b2col, b2row)

# Build the A0 and A1 matrices
A1 = block_matrix(b1col[0], b2col[0])
A0 = block_matrix(b1col[1], b2col[1])

# Now build the full Jacobian A
A = spla.aslinearoperator(sparse.kron(B1, M) + sparse.kron(B2, K))

qinit = np.zeros_like(mesh)
qinit[:] = np.cos(mesh/2)**4

# set up timeseries                                                                                                                                                                           
q = np.zeros(nt*nx)                                                                                                                                                                           
rhs = np.zeros_like(q)                                                                                                                                                                        
                                                                                                                                                                                              
q = q.reshape((nt, nx))                                                                                                                                                                       
rhs = rhs.reshape((nt, nx))                                                                                                                                                                   
                                                                                                                                                                                              
# initial guess is constant solution                                                                                                                                                          
q[:] = qinit[np.newaxis, :]                                                                                                                                                                   
rhs[0] -= A0.dot(qinit)                                                                                                                                                                       
                                                                                                                                                                                              
q = q.reshape(nx*nt)
rhs = rhs.reshape(nx*nt)

alpha = 1e-4
tol_range = np.logspace(-6, -1, 6)

P_exact = BlockCirculantLinearOperatorExact(b1col, b2col, block_matrix, nx, alpha)
q_exact, niterations_exact, residuals_exact, _ = modified_richardson(A, rhs, P_exact)
inexact_iterations = []
inexact_iterates = []
residuals = []
for tol in tol_range:
    P_inexact = BlockCirculantLinearOperatorInexact(b1col, b2col, block_matrix, nx, alpha, tol=tol)
    q_inexact, niterations_inexact, residuals_inexact, iterates_inexact = modified_richardson(A, rhs, P_inexact)
    inexact_iterations.append(niterations_inexact)
    inexact_iterates.append(iterates_inexact)
    residuals.append(residuals_inexact)

print(f"Exact iteration count: {niterations_exact}")

fig, ax = plt.subplots()
ax.plot(tol_range, inexact_iterations, color='b')
ax.axhline(niterations_exact, color='r', linestyle='--')

ax.set_xscale("log")

ax.set_xlabel(r"$\tau$")
ax.set_ylabel("Number of Iterations")

plt.show()

print("Inexact iterations:", inexact_iterations)
fig, ax = plt.subplots()
for res in residuals:
    ax.plot(res)
ax.set_xlabel("Iteration Number")
ax.set_ylabel('Residual')
ax.set_yscale("log")

plt.show()

# Fitting regression line to residuals to determine convergence rate
for res in residuals:
    r_0 = res[0]
    


# Plots using Errors instead of Residuals
errs = []
A_mat = sparse.kron(B1, M) + sparse.kron(B2, K)
exact = spla.spsolve(A_mat, rhs)
for iterates in inexact_iterates:
    errs_iterate = []
    for iter in iterates:
        errs_iterate.append(np.linalg.norm(iter - exact, 2))
    errs.append(errs_iterate)
fig, ax = plt.subplots()
for err in errs:
    ax.plot(err)
ax.set_xlabel("Iteration number")
ax.set_ylabel("Error")
ax.set_yscale("log")

plt.show()
