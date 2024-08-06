import numpy as np
from scipy.sparse import csc_array, coo_array
from Solverz.num_api.Array import Array
from Solverz.utilities.testing import assert_allclose_sparse


def test_Array():
    # input as number
    x1 = 1
    assert Array(x1, dim=1, dtype=float).__repr__() == 'array([1.])'
    assert Array(x1, dim=2, dtype=float).__repr__() == 'array([[1.]])'
    assert Array(x1, dim=2, sparse=False, dtype=float).__repr__() == 'array([[1.]])'
    assert_allclose_sparse(Array(x1, dim=2, sparse=True, dtype=float),
                           coo_array(np.array(x1).reshape((-1, 1)).astype(float)))

    # input as list
    x2 = [1, 2, 3]
    assert Array(x2, dim=1, dtype=float).__repr__() == 'array([1., 2., 3.])'
    assert Array(x2, dim=2, dtype=int).__str__() == '[[1]\n [2]\n [3]]'
    assert Array(x2, dim=2, sparse=False, dtype=float).__str__() == '[[1.]\n [2.]\n [3.]]'
    try:
        Array(x2, dim=1, sparse=True)
    except ValueError as e:
        assert e.args[0] == 'Cannot create sparse matrix with dim: 1'
    assert_allclose_sparse(Array(x2, dim=2, sparse=True, dtype=float),
                           coo_array(np.array(x2).reshape((-1, 1)).astype(float)))

    x2 = [[1, 0], [2, 9], [0, 3]]
    try:
        Array(x2, dim=1, dtype=float)
    except ValueError as e:
        assert e.args[0] == 'Input list dim 2 higher than dim set to be 1'

    assert Array(x2, dtype=float).__str__() == '[[1. 0.]\n [2. 9.]\n [0. 3.]]'

    # input as object
    x3 = [slice]
    try:
        Array(x3, dim=1, dtype=float)
    except TypeError as e:
        assert e.args[0] == 'Unsupported data type object'

    # input as csc_array
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    x4 = csc_array((data, indices, indptr), shape=(3, 3))
    try:
        Array(x4, dim=1)
    except ValueError as e:
        assert e.args[0] == 'csc_array input while dim set to be 1'
    try:
        Array(x4, sparse=False)
    except TypeError as e:
        assert e.args[0] == 'csc_array input while sparse arg set to be False'

    # input as numpy.ndarray
    x4 = np.array([[1, 0], [2, 9], [0, 3]])
    assert_allclose_sparse(Array(x4, sparse=True), coo_array(x4.astype(float)))
    try:
        Array(x4, dim=1, sparse=True)
    except ValueError as e:
        assert e.args[0] == 'Cannot create sparse matrix with dim: 1'

    assert Array(np.array([1.0, 2.0, 3.0]), dtype=int, dim=2).__str__() == '[[1]\n [2]\n [3]]'
    assert Array(np.array([1.0, 2.0, 3.0]), dtype=int, dim=1).__str__() == '[1 2 3]'
    assert_allclose_sparse(Array(np.array([1.0, 2.0, 3.0]),
                                 dtype=int,
                                 dim=2,
                                 sparse=True),
                           coo_array(np.array([1.0, 2.0, 3.0]).reshape((-1, 1)).astype(float)))
