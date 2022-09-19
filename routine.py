from __future__ import annotations

from eqn import Eqn, Equations
from var import Var, Vars
from typing import List, Union, Callable, Optional
from solverz_array import SolverzArray
from solver import nr_method


class Routine:

    def __init__(self,
                 equations: Equations,
                 var: Union[List[Var], Vars],
                 tol: float = 1e-9,
                 solver: Callable = nr_method):

        if not isinstance(var, Vars):
            self.y = Vars(var)
        else:
            self.y = var

        #  check if all variables are given in var
        for _name in equations.SYMBOLS.keys():
            if _name not in equations.PARAM.keys():
                if _name not in self.y.v.keys():
                    raise ValueError(f'Find undefined variable {_name}!')

        self.equations = equations
        self.tol = tol
        self.results = None
        self.method = solver

    def run(self):
        # TODO: SOLVE ME
        pass

    def __repr__(self):
        return f'Solving {self.equations.name} using {self.method} method'
