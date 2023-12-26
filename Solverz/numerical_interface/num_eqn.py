from typing import Callable, Dict

import numpy as np
from Solverz.auxiliary_service.address import Address
from Solverz.variable.variables import Vars, TimeVars


def parse_ae_v(y: np.ndarray, var_address: Address):
    return Vars(var_address, y)


def parse_dae_v(y: np.ndarray, var_address: Address):
    temp = Vars(var_address, y[0, :])
    temp = TimeVars(temp, y.shape[0])
    temp.array[:, :] = y
    return temp


class nAE:

    def __init__(self,
                 vsize,
                 g: Callable,
                 J: Callable,
                 p: Dict):
        self.vsize = vsize
        self.g = g
        self.J = J
        self.p = p


class nDAE:

    def __init__(self,
                 v_size,
                 M,
                 F: Callable,
                 J: Callable,
                 p: Dict):
        self.v_size = v_size
        self.M = M
        self.F = F
        self.J = J
        self.p = p
