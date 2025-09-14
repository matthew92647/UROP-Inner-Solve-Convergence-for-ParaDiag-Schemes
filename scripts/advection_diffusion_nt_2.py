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


alpha = 1e-4
nx = 128

lx = 2*pi
dx = lx/nx

theta = 0.55

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
K = (u/dx)*D - (nu/dx**2)*L

I = sparse.identity(K.shape[0], format='csc')

# Construct the matrix (I + dt*K)
A = I + dt*K

# Compute the inverse (dense) if small, or LU factorization
A_lu = spla.splu(A.tocsc())

# Apply the inverse to identity to get the inverse explicitly
A_inv = A_lu.solve(np.eye(K.shape[0]))

# Compute norms
norm_inf = np.linalg.norm(A_inv, np.inf)
norm_2 = np.linalg.norm(A_inv, 2)

print("|| (I + dt K)^-1 ||_inf =", norm_inf)
print("|| (I + dt K)^-1 ||_2   =", norm_2)

# Generate block matrices for different coefficients
def block_matrix(l1, l2):
    mat = l1*M + l2*K
    mat.solve = spla.factorized(mat.tocsc())
    return mat

np.random.seed(234)

taus = []
b_invs = []
ys = []

# Build the full B1 & B2 matrices
nt_range = np.logspace(4, 12, 9, base=2)
for nt in nt_range:
    nt = int(nt)
    print(nt)
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
    rhs = np.ones_like(q)

    q = q.reshape((nt, nx))
    rhs = rhs.reshape((nt, nx))

    # initial guess is constant solution
    q[:] = qinit[np.newaxis, :]
    rhs[0] -= A0.dot(qinit)

    q = q.reshape(nx*nt)
    rhs = rhs.reshape(nx*nt)

    vec_len = nt * nx
    b_rand = np.random.rand(vec_len)
    b_rand /= np.linalg.norm(b_rand, 2)

    P_exact = BlockCirculantLinearOperatorExact(b1col, b2col, block_matrix,
                                                nx, alpha)
    P_inexact = BlockCirculantLinearOperatorInexact(b1col, b2col, block_matrix,
                                                    nx, alpha)

    b = b_rand
    exact_solve = P_exact * b
    inexact_solve = P_inexact * b
    check_tol = P_inexact.global_tol
    max_norm = 0
    for Bi in P_inexact.blocks:
        Bi_lu = spla.splu(Bi.tocsc())
        I = np.eye(Bi.shape[0])  # noqa E741
        B_inv = Bi_lu.solve(I)
        norm = np.linalg.norm(B_inv, 2)
        if norm >= max_norm:
            max_norm = norm

    taus.append(check_tol)
    b_invs.append(max_norm)
    ys.append(np.linalg.norm(exact_solve - inexact_solve, 2)/check_tol)

y_bound = dt * alpha**(-(nt_range-1)/nt_range) * 1/(1 - alpha**(1/nt_range))
y_bound_approx = nt_range/alpha

fig, ax = plt.subplots()

ax.plot(nt_range, ys, label='Numerical')
ax.plot(nt_range, y_bound, 'r--', label='Theoretical bound')
ax.plot(nt_range, y_bound_approx, 'g--', label="Approximate Bound")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$N_t$", fontsize=12)
ax.set_ylabel(r"$\|\Delta w/\tau\|$", fontsize=12)
ax.set_title("Comparison of measured vs bound (2 norm)", fontsize=13)

ax.legend()

plt.show()

theory_b_inv = dt/(1 - np.array(alpha)**(1/nt_range))

fig, ax = plt.subplots()
ax.plot(nt_range, b_invs, label='Measured')
ax.plot(nt_range, theory_b_inv, 'r--', label='Theoretical bound')

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$N_t$", fontsize=12)
ax.set_ylabel(r"$\|B^{-1}\|$", fontsize=12)
ax.set_title("Comparison of measured vs bound (inverse matrix, 2 norm)", fontsize=13)

ax.legend()

plt.show()