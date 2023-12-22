import numpy as np
import tqdm
from typing import List, Union

from Solverz.solvers.aesolver import nr_method
from Solverz.symboli_algebra.symbols import Var
from Solverz.equation.equations import AE
from Solverz.variable.variables import TimeVars, Vars, combine_Vars, as_Vars


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
