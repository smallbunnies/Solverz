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
from .user_function_parser import load_my_module_paths

user_module_paths = load_my_module_paths()
if user_module_paths:
    print('User module detected.')
    import os, sys
    for path in user_module_paths:
        module_name = os.path.splitext(os.path.basename(path))[0]
        module_dir = os.path.dirname(path)

        sys.path.insert(0, module_dir)
        exec('import ' + module_name)
        module_dict[module_name] = globals()[module_name]


modules = [module_dict, 'numpy']
# We preserve the 'numpy' here in case one uses functions from sympy instead of from Solverz
__all__ = list(module_dict.keys())
