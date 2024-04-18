from numbers import Number

import pandas as pd
import numpy as np
import sympy as sp

from Solverz.sym_algebra.symbols import iVar, idx, Para
from Solverz.sym_algebra.functions import Abs
from Solverz.equation.eqn import Eqn, Ode
from Solverz.equation.equations import DAE, AE
from Solverz.variable.variables import combine_Vars, as_Vars

x = iVar('x', value=[1])
f = Eqn('f', x - 1)
f1 = f.NUM_EQN(1)
assert isinstance(f1, Number)

x = iVar('x', value=[1, 2])
f = Eqn('f', x - 1)
f1 = f.NUM_EQN(np.array([3, 4]))
assert isinstance(f1, np.ndarray)
assert f1.ndim == 1
