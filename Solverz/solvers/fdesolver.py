from typing import List, Union
from numbers import Number
import numpy as np
import tqdm

from Solverz.num_api.num_eqn import nFDAE, nAE
from Solverz.solvers.nlaesolver import nr_method
from Solverz.solvers.stats import Stats
from Solverz.solvers.option import Opt
from Solverz.solvers.parser import fdae_io_parser
from Solverz.solvers.solution import daesol


@fdae_io_parser
def fdae_solver(fdae: nFDAE,
                tspan: List | np.ndarray,
                y0: np.ndarray,
                opt: Opt = None,
                **kwargs):
    r"""
    The general solver of FDAE.

    Parameters
    ==========

    fdae : nFDAE
        Numerical FDAE object.

    tspan : List | np.ndarray
        An array specifying t0 and tend

    y0 : np.ndarray
        The initial values of variables

    opt : Opt
        The solver options, including:

        - step_size: 1e-3(default)|float
            The step size
        - ite_tol: 1e-8(default)|float
            The error tolerance of inner Newton iterations.

    kwargs : dict
        The value of y at previous time nodes. For example, if an FDAE uses $y_0$, $y_{-1}$ and $y_{-2}$, then we can
        call FDAE solver with

        .. code-block:: python

            sol = fdae_solver(mdl,
                              [0, 120],
                              y0,
                              Opt(step_size=60),
                              y1=y1,
                              y2=y2)

        where `y1` denotes $y_{-1}$ and `y2` denotes $y_{-2}$.

    Returns
    =======

    sol : daesol
        The daesol object.

    """
    stats = Stats(scheme='FDE solver')
    if opt is None:
        opt = Opt(stats=True)
    dt = opt.step_size
    tspan = np.array(tspan)
    T_initial = tspan[0]
    tend = tspan[-1]
    if (tend / dt) > 10000:
        nstep = np.ceil(tend / dt).astype(int) + 1000
    else:
        nstep = int(10000)
    nt0 = fdae.nstep - 1
    nt = nt0
    tt = T_initial
    t0 = tt
    uround = np.spacing(1.0)

    Y = np.zeros((nstep + nt0, y0.shape[0]))
    Y[nt, :] = y0
    T = np.zeros((nstep + nt0,))
    T[nt] = t0
    for j in range(1, nt0 + 1):
        Y[nt-j, :] = kwargs[f'y{j}']

    if opt.pbar:
        bar = tqdm.tqdm(total=tend - t0)

    done = False
    p = fdae.p
    while not done:

        if tt + dt >= tend:
            dt = tend - tt
        else:
            dt = np.minimum(dt, 0.5 * (tend - tt))

        if done:
            break

        ae = nAE(lambda y_, p_: fdae.F(t0 + dt, y_, p_, *[Y[nt - i, :] for i in range(fdae.nstep)]),
                 lambda y_, p_: fdae.J(t0 + dt, y_, p_, *[Y[nt - i, :] for i in range(fdae.nstep)]),
                 p)

        sol = nr_method(ae, y0, Opt(ite_tol=opt.ite_tol, stats=True))
        ynew = sol.y
        stats.ndecomp = stats.ndecomp + sol.stats.ndecomp
        stats.nfeval = stats.nfeval + sol.stats.nfeval
        if stats.nstep >= 100:
            print(f"FDAE solver broke at time={tt} due to non-convergence")
            break

        tt = tt + dt
        nt = nt + 1
        Y[nt] = ynew
        T[nt] = tt
        if opt.pbar:
            bar.update(dt)
        t0 = tt
        y0 = ynew

        if np.abs(tend - tt) < uround:
            done = True

    Y = Y[nt0:nt + 1]
    T = T[nt0:nt + 1]
    stats.nstep = nt
    if opt.pbar:
        bar.close()
    return daesol(T, Y, stats=stats)
