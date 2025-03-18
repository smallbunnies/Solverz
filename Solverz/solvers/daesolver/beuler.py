from Solverz.solvers.daesolver.utilities import *


@dae_io_parser
def backward_euler(dae: nDAE,
                   tspan: Union[List, np.ndarray],
                   y0: np.ndarray,
                   opt: Opt = None):
    r"""
    The fixed step backward Euler (implicit Euler) method.

    Parameters
    ==========

    dae : nDAE
        Numerical DAE object.

    tspan : List | np.ndarray
        An array specifying t0 and tend

    y0 : np.ndarray
        The initial values of variables

    opt : Opt
        The solver options, including:

        - step_size: 1e-3(default)|float
            The step size

    Returns
    =======

    sol : daesol
        The daesol object.


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Backward_Euler_method

    """
    stats = Stats(scheme='Backward Euler')
    if opt is None:
        opt = Opt(stats=True)
    dt = opt.step_size

    tspan = np.array(tspan)
    T_initial = tspan[0]
    T_end = tspan[-1]
    nt = 0
    tt = T_initial
    t0 = tt
    Nt = int((T_end-T_initial)/dt) + 1

    if opt.pbar:
        pbar = tqdm(total=T_end - T_initial)

    Y = np.zeros((Nt, y0.shape[0]))
    y0 = DaeIc(dae, y0, t0, opt.rtol)  # check and modify initial values
    Y[0, :] = y0
    T = np.zeros((Nt,))
    T[0] = t0

    p = dae.p
    while T_end - tt > abs(dt) / 10:
        My0 = dae.M @ y0
        ae = nAE(lambda y_, p_: dae.M @ y_ - My0 - dt * dae.F(t0 + dt, y_, p_),
                 lambda y_, p_: dae.M - dt * dae.J(t0 + dt, y_, p_),
                 p)

        sol = nr_method(ae, y0, Opt(stats=True))
        y1 = sol.y
        stats.ndecomp = stats.ndecomp + sol.stats.nstep
        stats.nfeval = stats.nfeval + sol.stats.nstep

        tt = tt + dt
        nt = nt + 1
        Y[nt] = y1
        T[nt] = tt
        if opt.pbar:
            pbar.update(T[nt] - T[nt - 1])
        y0 = y1
        t0 = tt

    Y = Y[0:nt + 1]
    T = T[0:nt + 1]
    if opt.pbar:
        pbar.close()
    stats.nstep = nt

    return daesol(T, Y, stats=stats)
