# parse modules from built-in custom functions
import numpy
import scipy
import warnings

import Solverz.num_api.custom_function as SolCF

modules = [{'SolCF': SolCF, 'np': numpy, 'sps': scipy.sparse}, 'numpy']
# We preserve the 'numpy' here in case one uses functions from sympy instead of from Solverz

# parse modules from museum
try:
    import SolMuseum.num_api as SolMF
    modules[0]['SolMF'] = SolMF
except ModuleNotFoundError as e:
    warnings.warn(f'Failed to import num api from SolMuseum: {e}')
