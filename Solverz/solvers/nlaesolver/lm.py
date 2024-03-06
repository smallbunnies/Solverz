from Solverz.solvers.nlaesolver.utilities import *


@ae_io_parser
def lm(eqn: nAE,
       y: np.ndarray,
       opt: Opt = None):
    if opt is None:
        opt = Opt(ite_tol=1e-8)

    tol = opt.ite_tol
    p = eqn.p

    # optimize.root func cannot handle callable jac that returns scipy.sparse.csc_array
    sol = optimize.root(lambda x: eqn.F(x, p), y, jac=lambda x: eqn.J(x, p).toarray(), method='lm', tol=tol)

    return aesol(sol.x, sol.njev)
