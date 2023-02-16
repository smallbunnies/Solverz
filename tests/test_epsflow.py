import numpy as np
import pandas as pd

from copy import deepcopy

from Solverz.eqn import Eqn
from Solverz.miscellaneous import derive_dhs_param_var
from Solverz.param import Param
from Solverz.routine import Routine
from Solverz.solver import *
from Solverz.var import Var
from Solverz.variables import Vars

Apq = Param(name='Apq')
Gpq = Param(name='Gpq')
Bpq = Param(name='Bpq')

E1 = Eqn(name='Active power',
         e_str='-Apq*p+Mat_Mul(Diagonal(Apq*e),Gpq*e-Bpq*f)',
         commutative=False)

# E = Equations(E1,
#               name='AC Power flow',
#               param=[Apq, Bpq, Gpq])


