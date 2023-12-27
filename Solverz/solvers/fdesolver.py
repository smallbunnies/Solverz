import numpy as np
import tqdm
from typing import List, Union

from Solverz.solvers.nlaesolver import nr_method, nr_method_numerical
from Solverz.symboli_algebra.symbols import Var
from Solverz.equation.equations import AE
from Solverz.numerical_interface.num_eqn import nFDAE, nAE
from Solverz.variable.variables import TimeVars, Vars, combine_Vars, as_Vars
from Solverz.solvers.stats import Stats
from Solverz.solvers.option import Opt


def fde_solver(fde: AE,
               u: Vars,
               tspan: Union[List, np.ndarray],
               dt,
               tol=1e-5):
    # tspan = np.array(tspan)
    T_initial = tspan[0]
    t = Var('t', value=T_initial)
    u = combine_Vars(u, as_Vars(t))
    T_end = tspan[-1]
    # i = 0
    tt = T_initial
    u = TimeVars(u, length=int(T_end / dt) + 1)
    u1 = u[0]  # x_{i+1}
    fde.update_param('dt', dt)
    u0 = u1.derive_alias('0')  # x_{i}

    for i in range(int(T_end / dt)):
        fde.update_param(u0)
        # fde.update_param('t', t + dt)
        fde.update_param('t0', tt)
        u1 = nr_method(fde, u1, tol=tol)
        u[i + 1] = u1
        tt = tt + dt
        # if pbar:
        #     bar.update(dt)
        u0.array[:] = u1.array[:]

    return u


def fdae_solver_numerical(fdae: nFDAE,
                          u0: np.ndarray,
                          tspan: Union[List, np.ndarray],
                          dt,
                          tol=1e-5):
    stats = Stats(scheme='FDE solver')

    tspan = np.array(tspan)
    T_initial = tspan[0]
    tend = tspan[-1]
    nt = 0
    tt = T_initial
    t0 = tt
    uround = np.spacing(1.0)

    u = np.zeros((10000, fdae.v_size))
    u[0, :] = u0
    T = np.zeros((10000,))

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
            ae = nAE(fdae.v_size,
                     lambda y, p: fdae.F(t0 + dt, y, p, u0),
                     lambda y, p: fdae.J(t0 + dt, y, p, u0),
                     p)
        else:
            raise ValueError("Multistep FDAE not implemented!")
        
        u1, ite = nr_method_numerical(ae, u0, tol=tol, stats=True)
        stats.ndecomp = stats.ndecomp + ite
        stats.nfeval = stats.nfeval + ite

        tt = tt + dt
        nt = nt + 1
        u[nt] = u1
        T[nt] = tt
        t0 = tt
        u0 = u1

        if np.abs(tend - tt) < uround:
            done = True

    u = u[0:nt+1]
    T = T[0:nt+1]
    stats.nstep = nt

    return T, u, stats
