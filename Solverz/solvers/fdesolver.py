from typing import List, Union
from numbers import Number
import numpy as np
import tqdm

from Solverz.numerical_interface.num_eqn import nFDAE, nAE
from Solverz.solvers.nlaesolver import nr_method
from Solverz.solvers.stats import Stats
from Solverz.solvers.option import Opt


def fdae_solver(fdae: nFDAE,
                tspan: Union[List, np.ndarray],
                u0: np.ndarray,
                dt,
                opt: Opt = None):
    stats = Stats(scheme='FDE solver')
    if opt is None:
        opt = Opt(stats=True)

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
    if opt.pbar:
        bar = tqdm.tqdm(total=tend)

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

        u1, ite = nr_method(ae, u0, Opt(ite_tol=opt.ite_tol, stats=True))
        stats.ndecomp = stats.ndecomp + ite
        stats.nfeval = stats.nfeval + ite + 1

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
    return T, u, stats


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
