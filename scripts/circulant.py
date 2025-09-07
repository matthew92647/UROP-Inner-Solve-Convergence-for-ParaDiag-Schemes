from scipy import sparse
from scipy.sparse import linalg as spla
import numpy as np
from scipy.fft import fft, ifft

# Finite difference spatial discretisations


def gradient_stencil(grad, order):
    '''
    Return the centred stencil for the `grad`-th gradient
    of order of accuracy `order`
    '''
    return {
        1: {  # first gradient
            2: np.array([-1/2, 0, 1/2]),
            4: np.array([1/12, -2/3, 0, 2/3, -1/12]),
            6: np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
        },
        2: {  # second gradient
            2: np.array([1, -2, 1]),
            4: np.array([-1/12, 4/3, -5/2, 4/3, -1/12]),
            6: np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
        },
        4: {  # fourth gradient
            2: np.array([1,  -4, 6, -4, 1]),
            4: np.array([-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6]),
            6: np.array([7/240, -2/5, 169/60, -122/15, 91/8, -122/15, 169/60, -2/5, 7/240])  # noqa: E501
        }
    }[grad][order]


def sparse_circulant(stencil, n):
    '''
    Return sparse scipy matrix from finite difference
    stencil on a periodic grid of size n.
    '''
    if len(stencil) == 1:
        return sparse.spdiags([stencil[0]*np.ones(n)], 0)

    # extend stencil to include periodic overlaps
    ns = len(stencil)
    noff = (ns-1)//2
    pstencil = np.zeros(ns+2*noff)

    pstencil[noff:-noff] = stencil
    pstencil[:noff] = stencil[noff+1:]
    pstencil[-noff:] = stencil[:noff]

    # constant diagonals of stencil entries
    pdiags = np.tile(pstencil[:, np.newaxis], n)

    # offsets for inner domain and periodic overlaps
    offsets = np.zeros_like(pstencil, dtype=int)

    offsets[:noff] = [-n+1+i for i in range(noff)]
    offsets[noff:-noff] = [-noff+i for i in range(2*noff+1)]
    offsets[-noff:] = [n-noff+i for i in range(noff)]

    return sparse.spdiags(pdiags, offsets)


class BlockCirculantLinearOperatorExact(spla.LinearOperator):
    def __init__(self, b1col, b2col, block_matrix, nx, alpha=1):
        self.nt = len(b1col)
        self.nx = nx
        self.dim = self.nt*self.nx
        self.shape = tuple((self.dim, self.dim))
        self.dtype = b1col.dtype

        self.gamma = alpha**(np.arange(self.nt)/self.nt)

        eigvals1 = fft(b1col*self.gamma, norm='backward')
        eigvals2 = fft(b2col*self.gamma, norm='backward')
        eigvals = zip(eigvals1, eigvals2)

        self.blocks = tuple((block_matrix(l1, l2)
                             for l1, l2 in eigvals))

    def _to_eigvecs(self, v):
        y = np.matmul(np.diag(self.gamma), v)
        return fft(y, axis=0)

    def _from_eigvecs(self, v):
        y = ifft(v, axis=0)
        return np.matmul(np.diag(1/self.gamma), y)

    def _block_solve(self, v):
        for i in range(self.nt):
            v[i] = self.blocks[i].solve(v[i])
        return v

    def _matvec(self, v):
        y = v.reshape((self.nt, self.nx))
        y = self._to_eigvecs(y)
        y = self._block_solve(y)
        y = self._from_eigvecs(y)
        return y.reshape(self.dim).real


class BlockCirculantLinearOperatorInexact(spla.LinearOperator):
    def __init__(self, b1col, b2col, block_matrix, nx, alpha=1, tol=0.001):
        self.nt = len(b1col)
        self.nx = nx
        self.dim = self.nt*self.nx
        self.shape = tuple((self.dim, self.dim))
        self.dtype = b1col.dtype
        self.tol = tol
        self.gamma = alpha**(np.arange(self.nt)/self.nt)

        eigvals1 = fft(b1col*self.gamma, norm='backward')
        eigvals2 = fft(b2col*self.gamma, norm='backward')
        eigvals = zip(eigvals1, eigvals2)

        self.blocks = tuple((block_matrix(l1, l2)
                             for l1, l2 in eigvals))
        self.global_tol = None

    def _to_eigvecs(self, v):
        y = np.matmul(np.diag(self.gamma), v)
        return fft(y, axis=0)

    def _from_eigvecs(self, v):
        y = ifft(v, axis=0)
        return np.matmul(np.diag(1/self.gamma), y)

    def _block_solve(self, v):  # method for inexact block solve
        for i in range(self.nt):
            vi = v[i]
            vi_approx, exit_code = spla.gmres(self.blocks[i], vi,
                                              rtol=self.tol)
            v[i] = vi_approx
        return v

    def _matvec(self, v):
        y = v.reshape((self.nt, self.nx))
        y = self._to_eigvecs(y)
        y_exact = y.copy()
        # compute the exact inner solve for each block
        y = self._block_solve(y)  # compute inexact solve for each block
        res_blocks = []
        for i in range(self.nt):
            res_blocks.append(np.linalg.norm(self.blocks[i].dot(y[i]) - y_exact[i], np.inf)/np.linalg.norm(y_exact[i], np.inf))
        self.res_blocks = res_blocks
        res_blocks = []
        for i in range(self.nt):
            res_blocks.append(self.blocks[i].dot(y[i]) - y_exact[i])
        res = np.vstack(res_blocks)
        rel_res = np.linalg.norm(res, 2) / np.linalg.norm(y_exact, 2)
        rel_res_inf = np.linalg.norm(res, np.inf) / np.linalg.norm(y_exact, np.inf)
        self.global_tol = rel_res
        self.global_tol_inf = rel_res_inf
        y = self._from_eigvecs(y)
        return y.reshape(self.dim).real
