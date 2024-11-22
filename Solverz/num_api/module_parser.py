# parse modules from built-in custom functions
import numpy
import scipy
import warnings

import Solverz.num_api.custom_function as SolCF
import numpy as np
import scipy.sparse as sps
from numba import njit

module_dict = {'SolCF': SolCF, 'np': np, 'sps': sps, 'njit': njit}

# parse modules from museum
try:
    import SolMuseum.num_api as SolMF
    module_dict['SolMF'] = SolMF
except ModuleNotFoundError as e:
    warnings.warn(f'Failed to import num api from SolMuseum: {e}')

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
