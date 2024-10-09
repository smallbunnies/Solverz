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
                u0: np.ndarray,
                opt: Opt = None):
    r"""
    The general solver of FDAE.

    Parameters
    ==========

    fdae : nFDAE
        Numerical FDAE object.

    tspan : List | np.ndarray
        An array specifying t0 and tend

    u0 : np.ndarray
        The initial values of variables

    opt : Opt
        The solver options, including:

        - step_size: 1e-3(default)|float
            The step size
        - ite_tol: 1e-8(default)|float
            The error tolerance of inner Newton iterations.

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
    nt = 0
    tt = T_initial
    t0 = tt
    uround = np.spacing(1.0)

    u = np.zeros((nstep, u0.shape[0]))
    u[0, :] = u0
    T = np.zeros((nstep,))
    T[0] = t0
    if opt.pbar:
        bar = tqdm.tqdm(total=tend-t0)

    done = False
    p = fdae.p
    while not done:

        if tt + dt >= tend:
            dt = tend - tt
        else:
            dt = np.minimum(dt, 0.5 * (tend - tt))

        if done:
            break

        if fdae.nstep == 1:
            ae = nAE(lambda y_, p_: fdae.F(t0 + dt, y_, p_, u0),
                     lambda y_, p_: fdae.J(t0 + dt, y_, p_, u0),
                     p)
        else:
            raise NotImplementedError("Multistep FDAE not implemented!")

        sol = nr_method(ae, u0, Opt(ite_tol=opt.ite_tol, stats=True))
        u1 = sol.y
        stats.ndecomp = stats.ndecomp + sol.stats.ndecomp
        stats.nfeval = stats.nfeval + stats.nfeval
        if stats.nstep >= 100:
            print(f"FDAE solver broke at time={tt} due to non-convergence")
            break

        tt = tt + dt
        nt = nt + 1
        u[nt] = u1
        T[nt] = tt
        if opt.pbar:
            bar.update(dt)
        t0 = tt
        u0 = u1

        if np.abs(tend - tt) < uround:
            done = True

    u = u[0:nt + 1]
    T = T[0:nt + 1]
    stats.nstep = nt
    if opt.pbar:
        bar.close()
    return daesol(T, u, stats=stats)


def fdae_ss_solver(fdae: nFDAE,
                   u0: np.ndarray,
                   dt,
                   T0: Number = 0,
                   opt: Opt = None):
    stats = Stats(scheme='FDE solver')
    if opt is None:
        opt = Opt()

    nstep = 10000
    nt = 0
    tt = T0
    t0 = tt

    u = np.zeros((nstep, u0.shape[0]))
    u[0, :] = u0

    done = False
    p = fdae.p
    dev0 = 1
    while not done:

        if done:
            break

        if fdae.nstep == 1:
            ae = nAE(lambda y_, p_: fdae.F(t0 + dt, y_, p_, u0),
                     lambda y_, p_: fdae.J(t0 + dt, y_, p_, u0),
                     p)
        else:
            raise NotImplementedError("Multistep FDAE not implemented!")

        u1, ite = nr_method(ae, u0, tol=opt.ite_tol, stats=True)
        stats.ndecomp = stats.ndecomp + ite
        stats.nfeval = stats.nfeval + ite + 1

        tt = tt + dt
        nt = nt + 1
        u[nt] = u1

        du = u1 - u0
        args = u1 != 0
        dev1 = np.max(np.abs(du[args] / u1[args]))
        if dev1 < 1e-7:
            done = True

        u0 = u1
        t0 = tt
        # if dev1 <= dev0:
        #     dt = dt * 1.1
        dev0 = dev1

    u = u[0:nt + 1]
    stats.nstep = nt

    return u, stats
