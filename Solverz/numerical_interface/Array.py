from typing import Union

import numpy as np
from numbers import Number
from scipy.sparse import csc_array


def type_checker(dtype):
    if not np.issubdtype(dtype, np.integer) and not np.issubdtype(dtype, np.floating) and not np.issubdtype(dtype, np.complexfloating):
        raise TypeError(f"Unsupported data type {dtype}")


def Array(array: Union[np.ndarray, csc_array, list, Number],
          dim=2,
          sparse=False,
          dtype=float) -> Union[np.ndarray, csc_array]:
    type_checker(dtype)
    if dim < 2 and sparse:
        raise ValueError(f"Cannot create sparse matrix with dim: {dim}")

    # initialize and check dtype in case of non-int/float dtypes
    if isinstance(array, Number):
        temp = np.array(array, dtype=dtype)
    elif isinstance(array, list):
        temp = np.array(array)
        type_checker(temp.dtype)
        if temp.ndim > dim:
            raise ValueError(f"Input list dim {temp.ndim} higher than dim set to be {dim}")
        temp = temp.astype(dtype)
    elif isinstance(array, np.ndarray):
        type_checker(array.dtype)
        temp = array.astype(dtype)
        if temp.ndim > dim:
            raise ValueError(f"Input numpy.ndarray dim {temp.ndim} higher than dim set to be {dim}")
    elif isinstance(array, csc_array):
        type_checker(array.dtype)
        temp = array.astype(dtype)
        if dim != 2:
            raise ValueError(f"csc_array input while dim set to be {dim}")
        if not sparse:
            raise TypeError(f"csc_array input while sparse arg set to be False")
    else:
        raise TypeError(f"Unsupported array type {type(array)}")

    # reshape and sparsify
    if dim == 1:
        # This can only be np.ndarray
        # dim of csc_array must be 2
        return temp.reshape((-1,))
    else:
        # np.ndarray or csc_array
        if temp.ndim < 2:
            temp = temp.reshape((-1, 1))
        if sparse:
            return csc_array(temp)
        else:
            return temp
