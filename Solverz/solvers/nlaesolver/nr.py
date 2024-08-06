from Solverz.solvers.nlaesolver.utilities import *


@ae_io_parser
def nr_method(eqn: nAE,
              y: np.ndarray,
              opt: Opt = None):
    r"""
    The Newton-Raphson method.

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

    .. [1] https://en.wikipedia.org/wiki/Newton%27s_method

    """
    if opt is None:
        opt = Opt()

    tol = opt.ite_tol
    p = eqn.p
    df = eqn.F(y, p)

    stats = Stats('Newton')
    stats.nfeval += 1

    # main loop
    while max(abs(df)) > tol:
        y = y - solve(eqn.J(y, p), df)
        stats.ndecomp += 1
        df = eqn.F(y, p)
        stats.nfeval += 1
        stats.nstep += 1

        if stats.nstep >= 100:
            print(f"Cannot converge within 100 iterations. Deviation: {max(abs(df))}!")
            stats.succeed = False
            break

    if np.any(np.isnan(y)):
        stats.succeed = False

    return aesol(y, stats)
