# parse modules from built-in custom functions
import Solverz.num_api.custom_function as SolCF
import numpy, scipy
modules = [{'SolCF': SolCF, 'np': numpy, 'sps': scipy.sparse}, 'numpy']
# We preserve the 'numpy' here in case one uses functions from sympy instead of from Solverz

# parse modules from museum
try:
    import SolMuseum.num_api as SolMF
    modules[0]['SolMF'] = SolMF
except ImportError as e:
    warnings.warn(f'Failed to import num api from SolMuseum: {e}')
