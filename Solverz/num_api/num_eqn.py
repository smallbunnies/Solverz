from typing import Callable, Dict

import numpy as np

from Solverz.equation.param import TimeSeriesParam
from Solverz.variable.variables import Vars, TimeVars, combine_Vars


class nAE:

    def __init__(self,
                 F: Callable,
                 J: Callable,
                 p: Dict):
        self.F = F
        self.J = J
        self.p = p


class nFDAE:

    def __init__(self,
                 F: callable,
                 J: callable,
                 p: dict,
                 nstep: int = 0):
        self.F = F
        self.J = J
        self.p = p
        self.nstep = nstep


class nDAE:

    def __init__(self,
                 M,
                 F: Callable,
                 J: Callable,
                 p: Dict):
        self.M = M
        self.F = F
        self.J = J
        self.p = p
