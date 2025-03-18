from Solverz.solvers.nlaesolver.utilities import *


@ae_io_parser
def lm(eqn: nAE,
       y: np.ndarray,
       opt: Opt = None):
    r"""
    The Levenberg-Marquardt method by minpack, wrapped in scipy.

    .. warning::
        Note that this function uses only dense Jacobian.

    Parameters
    ==========

    eqn : nAE
        Numerical AE object.

    y : np.ndarray
        The initial values of variables

    opt : Opt
        The solver options, including:

        - ite_tol: 1e-5(default)|float
            The iteration error tolerance.

    Returns
    =======

    sol : aesol
        The aesol object.

    References
    ==========

    .. [1] More, Jorge J., Burton S. Garbow, and Kenneth E. Hillstrom. 1980. User Guide for MINPACK-1.
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html

    """
    if opt is None:
        opt = Opt()

    stats = Stats('Levenberg–Marquardt')

    tol = opt.ite_tol
    p = eqn.p

    # optimize.root func cannot handle callable jac that returns scipy.sparse.csc_array
    sol = optimize.root(lambda x: eqn.F(x, p), 
                        y, 
                        jac=lambda x: eqn.J(x, p).toarray(), 
                        method='lm', 
                        tol=tol,
                        options={'maxiter': opt.max_it})
    dF = eqn.F(sol.x, eqn.p)
    if np.max(np.abs(dF)) < tol:
        stats.succeed = True
    stats.nfeval = sol.nfev

    return aesol(sol.x, stats)
