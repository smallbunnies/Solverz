# parse modules from built-in custom functions
import numpy
import scipy
import warnings
from importlib.metadata import entry_points

import Solverz.num_api.custom_function as SolCF
import numpy as np
import scipy.sparse as sps
from numba import njit

module_dict = {'SolCF': SolCF, 'np': np, 'sps': sps, 'njit': njit}

# Discover and load num_api plugins registered via the 'solverz.num_api' entry point group.
# Third-party packages (e.g. SolMuseum) can register themselves without Solverz knowing their
# name by adding an entry point in their pyproject.toml:
#
#   [project.entry-points."solverz.num_api"]
#   SolMF = "SolMuseum.num_api"
#
for ep in entry_points(group='solverz.num_api'):
    try:
        module = ep.load()
        module_dict[ep.name] = module
    except Exception as e:
        warnings.warn(f'Failed to load num_api plugin {ep.name!r}: {e}')

# parse user defined functions

try:
    import myfunc
    print('User module detected.')
    module_dict['myfunc'] = myfunc
except ModuleNotFoundError as e:
    pass

modules = [module_dict, 'numpy']
# We preserve the 'numpy' here in case one uses functions from sympy instead of from Solverz
__all__ = list(module_dict.keys())
