from typing import List, Union

import numpy as np
import tqdm

from Solverz.numerical_interface.num_eqn import nFDAE, nAE
from Solverz.solvers.nlaesolver import nr_method
from Solverz.solvers.stats import Stats


def fdae_solver(fdae: nFDAE,
                tspan: Union[List, np.ndarray],
                u0: np.ndarray,
                dt,
                tol=1e-5,
                pbar=False):
    stats = Stats(scheme='FDE solver')

    tspan = np.array(tspan)
    T_initial = tspan[0]
    tend = tspan[-1]
    if (tend/dt) > 10000:
        nstep = np.ceil(tend/dt).astype(int)+1000
    else:
        nstep = int(10000)
    nt = 0
    tt = T_initial
    t0 = tt
    uround = np.spacing(1.0)

    u = np.zeros((nstep, u0.shape[0]))
    u[0, :] = u0
    T = np.zeros((nstep,))
    if pbar:
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

        u1, ite = nr_method(ae, u0, tol=tol, stats=True)
        stats.ndecomp = stats.ndecomp + ite
        stats.nfeval = stats.nfeval + ite + 1

        tt = tt + dt
        nt = nt + 1
        u[nt] = u1
        T[nt] = tt
        if pbar:
            bar.update(dt)
        t0 = tt
        u0 = u1

        if np.abs(tend - tt) < uround:
            done = True

    u = u[0:nt + 1]
    T = T[0:nt + 1]
    stats.nstep = nt
    if pbar:
        bar.close()
    return T, u, stats
