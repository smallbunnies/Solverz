from Solverz.solvers.nlaesolver.utilities import *


@ae_io_parser
def nr_method(eqn: nAE,
              y: np.ndarray,
              opt: Opt = None):
    if opt is None:
        opt = Opt(ite_tol=1e-8)

    tol = opt.ite_tol
    p = eqn.p
    df = eqn.F(y, p)
    ite = 0
    # main loop
    while max(abs(df)) > tol:
        ite = ite + 1
        y = y - solve(eqn.J(y, p), df)
        df = eqn.F(y, p)
        if ite >= 100:
            print(f"Cannot converge within 100 iterations. Deviation: {max(abs(df))}!")
            break

    return aesol(y, ite)
