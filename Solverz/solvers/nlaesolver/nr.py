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

    stats = Stats('Newton')
    tol = opt.ite_tol
    p = eqn.p
    df = eqn.F(y, p)
    stats.nfeval += 1
    # Reuse the KLU symbolic ordering across the iterations and across the
    # calls: the Jacobian pattern of an nAE is fixed, so only its values change
    # between Newton steps, and a repeated solve of the same equations must not
    # pay the symbolic analysis and the row matching again. The holder lives on
    # the nAE (laesolver.model_cache); a pattern change is detected there.
    cache = model_cache(eqn)

    # main loop
    while np.max(np.abs(df)) > tol:

        if stats.nstep > opt.max_it:
            print(f"Cannot converge within 100 iterations. Deviation: {np.max(np.abs(df))}!")
            break

        stats.nstep += 1

        y = y - solve(eqn.J(y, p), df, cache=cache)
        stats.nJeval += 1
        stats.ndecomp += 1
        stats.nsolve += 1
        df = eqn.F(y, p)
        stats.nfeval += 1

    if np.max(np.abs(df)) < tol:
        stats.succeed = True
    else:
        J = eqn.J(y, p)
        import scipy.sparse as _sps
        Jd = J.data if _sps.issparse(J) else np.asarray(J).ravel()
        print(f"[diag] nr_method failed: nstep {stats.nstep}, max|F| {np.max(np.abs(df))}, nan in F {int(np.isnan(df).sum())}, "
              f"J shape {J.shape} nnz {getattr(J, 'nnz', Jd.size)} nan in J {int(np.isnan(Jd).sum())} "
              f"y nan {int(np.isnan(y).sum())} y max {np.nanmax(np.abs(y))}", flush=True)
        if _sps.issparse(J):
            Jc = J.tocsc(); empty_rows = int((np.diff(J.tocsr().indptr) == 0).sum()); empty_cols = int((np.diff(Jc.indptr) == 0).sum())
            print(f"[diag] J empty rows {empty_rows} empty cols {empty_cols}; row/col ranges {J.tocoo().row.min()}..{J.tocoo().row.max()} / {J.tocoo().col.min()}..{J.tocoo().col.max()}", flush=True)

    return aesol(y, stats)
