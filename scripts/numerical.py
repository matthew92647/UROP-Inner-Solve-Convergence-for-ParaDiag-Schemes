import numpy as np


def modified_richardson(A, b, P, omega=1.0, tol=1e-5, maxiter=1000):
    """
    Modified Richardson iteration for Ax = b with preconditioner P.
    """
    x = np.zeros_like(b)
    r = b - A @ x
    iterates = [x.copy()]
    residuals = [np.linalg.norm(P@r, np.inf)/np.linalg.norm(b, np.inf)]
    for k in range(maxiter):
        x += omega * (P @ r)
        iterates.append(x.copy())
        r = b - A @ x
        pre_res = P @ r
        residuals.append(np.linalg.norm(pre_res, np.inf)/np.linalg.norm(b, np.inf))
        if np.linalg.norm(P @ r, np.inf)/np.linalg.norm(b, np.inf) < tol:
            return x, k+1, residuals, iterates
    return x, maxiter, residuals, iterates
