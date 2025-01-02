import numpy as np
from numpy.linalg import norm
from Solverz.num_api.num_eqn import nDAE
from Solverz.solvers.laesolver import solve

from scipy.sparse.linalg import svds

def getyp0(dae: nDAE, y0: np.ndarray, t0):

    M = dae.M
    p = dae.p
    F0 = dae.F(t0, y0, p)
    DiffEqn, DiffVar = M.nonzero()

    yp0 = np.zeros_like(y0)

    nonzero_rows = set(DiffEqn)
    nonzero_cols = set(DiffVar)
    Mp = M[list(nonzero_rows), :][:, list(nonzero_cols)]
    yp0[DiffVar] = solve(Mp, F0[DiffEqn])
    return yp0


def DaeIc(dae: nDAE, y0: np.ndarray, t0, rtol):
    """
    Check and solve for more consistent initial values of algebraic variables in DAEs
    Only supports mass matrix with one-entry in each column/row.
    """
    M = dae.M
    p = dae.p
    F0 = dae.F(t0, y0, p)
    DiffEqn, DiffVar = M.nonzero()
    AlgVar = np.setdiff1d(np.arange(y0.shape[0]), DiffVar)
    AlgEqn = np.setdiff1d(np.arange(y0.shape[0]), DiffEqn)
    if norm(F0[AlgEqn]) <= 1e-6:
        return y0
    else:
        F = lambda y_: dae.F(t0, y_, p)
        J = lambda y_: dae.J(t0, y_, p)
        y = y0
        for n in range(15):
            Jn = J(y)[np.ix_(AlgEqn, AlgVar)]
            Fn = F(y)
            dY = solve(Jn, -Fn[AlgEqn])
            nz_idx = np.abs(y[AlgVar]) > np.spacing(t0)  # in case some variables are too small
            res = norm(dY[nz_idx]/y[AlgVar][nz_idx])
            # week line search with affine invariant test
            lam = 1
            ynew = y.copy()
            for probe in range(3):
                ynew[AlgVar] = y[AlgVar] + lam * dY
                Fnew = F(ynew)
                if norm(Fnew[AlgEqn]) <= 1e-5 * rtol:
                    return ynew
                dYnew = solve(J(ynew)[np.ix_(AlgEqn, AlgVar)], Fnew[AlgEqn])
                resnew = norm(dYnew[nz_idx]/y[AlgVar][nz_idx])
                if resnew < 0.9 * res:
                    break
                else:
                    lam = 0.5 * lam
            # Ynorm = np.max([norm(y[AlgVar]), norm(ynew[AlgVar])])
            # if Ynorm == 0:
            #     Ynorm = np.spacing()
            y = ynew
            if resnew <= 1e-3 * rtol:
                return ynew

    raise ValueError("Need Better y0")
