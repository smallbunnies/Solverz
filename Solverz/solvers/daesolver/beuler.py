from Solverz.solvers.daesolver.utilities import *


@dae_io_parser
def backward_euler(dae: nDAE,
                   tspan: Union[List, np.ndarray],
                   y0: np.ndarray,
                   opt: Opt = None):
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

    y = np.zeros((10000, y0.shape[0]))
    y[0, :] = y0
    T = np.zeros((10000,))

    p = dae.p
    while abs(tt - T_end) > abs(dt) / 10:
        My0 = dae.M @ y0
        ae = nAE(lambda y_, p_: dae.M @ y_ - My0 - dt * dae.F(t0 + dt, y_, p_),
                 lambda y_, p_: dae.M - dt * dae.J(t0 + dt, y_, p_),
                 p)

        y1, ite = nr_method(ae, y0, Opt(stats=True))
        stats.ndecomp = stats.ndecomp + ite
        stats.nfeval = stats.nfeval + ite

        tt = tt + dt
        nt = nt + 1
        y[nt] = y1
        T[nt] = tt
        y0 = y1
        t0 = tt

    y = y[0:nt + 1]
    T = T[0:nt + 1]
    stats.nstep = nt

    return daesol(T, y, stats=stats)
