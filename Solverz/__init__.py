from Solverz.equation.eqn import Eqn, Ode, HyperbolicPde
from Solverz.equation.equations import AE, FDAE, DAE
from Solverz.equation.param import Param, IdxParam, TimeSeriesParam
from Solverz.sym_algebra.symbols import idx, Para, iVar, iAliasVar
from Solverz.sym_algebra.functions import Sign, Abs, transpose, exp, Diag, Mat_Mul, sin, cos, Min, AntiWindUp, Saturation
from Solverz.num_api.custom_function import minmod_flag, minmod
from Solverz.variable.variables import Vars, TimeVars, as_Vars
from Solverz.solvers import *
from Solverz.code_printer import made_numerical, module_printer
from Solverz.utilities.io import save, load, save_result
from Solverz.utilities.profile import count_time
from Solverz.variable.ssymbol import Var, AliasVar
from Solverz.model.basic import Model

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("Solverz")
except PackageNotFoundError:
    # package is not installed
    pass
