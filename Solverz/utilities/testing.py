import numpy as np
from scipy.sparse import coo_array, sparray


def assert_allclose_sparse(a, b, rtol=1e-7, atol=0):
    if not isinstance(a, sparray):
        raise TypeError(f"a should be scipy.sparray, instead of {type(a)}")
    if not isinstance(b, sparray):
        raise TypeError(f"b should be scipy.sparray, instead of {type(b)}")
    a_ = coo_array(a)
    b_ = coo_array(b)
    np.testing.assert_allclose(a_.data, b_.data, rtol, atol)
    np.testing.assert_allclose(a_.row, b_.row, rtol, atol)
    np.testing.assert_allclose(a_.col, b_.col, rtol, atol)
