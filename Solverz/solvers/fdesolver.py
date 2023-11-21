import numpy as np
import tqdm
from typing import List, Union

from Solverz.solvers.aesolver import nr_method
from Solverz.event import Event
from Solverz.equation.equations import AE
from Solverz.variable.variables import TimeVars, Vars


def fde_solver(fde: AE,
               u: Vars,
               tspan: Union[List, np.ndarray],
               dt,
               dx,
               L,
               tol=1e-5,
               event: Event = None):
    # tspan = np.array(tspan)
    # T_initial = tspan[0]
    T_end = tspan[-1]
    # i = 0
    # t = T_initial
    u = TimeVars(u, length=int(T_end/dt)+1)
    u1 = u[0]  # x_{i+1}
    fde.update_param('dt', dt)
    fde.update_param('dx', dx)
    fde.update_param('M', L/dx)
    u0 = u1.derive_alias('0')  # x_{i}

    for i in range(int(T_end/dt)):

        # if event:
        #     fde.update_param(event, t)

        fde.update_param(u0)
        # fde.update_param('t', t + dt)
        # fde.update_param('t0', t)
        if event:
            fde.update_param(event, i*dt)
        u1 = nr_method(fde, u1, tol=tol)
        u[i + 1] = u1
        # if pbar:
        #     bar.update(dt)
        u0.array[:] = u1.array[:]

    return u
