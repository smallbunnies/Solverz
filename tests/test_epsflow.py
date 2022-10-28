import numpy as np
import pandas as pd

from copy import deepcopy

from core.eqn import Eqn
from core.miscellaneous import derive_dhs_param_var
from core.param import Param
from core.routine import Routine
from core.solver import *
from core.var import Var
from core.variables import Vars

Apq = Param(name='Apq')
Gpq = Param(name='Gpq')
Bpq = Param(name='Bpq')

E1 = Eqn(name='Active power',
         e_str='-Apq*p+Mat_Mul(Diagonal(Apq*e),Gpq*e-Bpq*f)',
         commutative=False)

E = Equations(E1,
              name='AC Power flow',
              param=[Apq, Bpq, Gpq])


