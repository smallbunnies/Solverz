from __future__ import annotations

from typing import Union, List

import numpy as np
from numpy import abs, linalg
from scipy.sparse import diags_array
from tqdm import tqdm

from Solverz.equation.equations import DAE
from Solverz.num_api.num_eqn import nDAE, nAE
from Solverz.solvers.laesolver import lu_decomposition
from Solverz.solvers.nlaesolver import nr_method
from Solverz.solvers.option import Opt
from Solverz.solvers.parser import dae_io_parser
from Solverz.solvers.stats import Stats
from Solverz.variable.variables import TimeVars
from Solverz.solvers.solution import daesol
from Solverz.solvers.daesolver.daeic import DaeIc, getyp0
from Solverz.num_api.numjac import numjac
