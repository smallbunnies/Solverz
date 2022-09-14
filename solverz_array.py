import numpy as np
from numbers import Number
import numpy.lib.mixins
from typing import Union
from functools import reduce

Np_Mapping = {}


class SolverzArray(np.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self,
                 array: Union[list, np.ndarray],
                 dtype='float'):

        if isinstance(array, list):
            self.dtype = dtype
            self.array = np.array(array, dtype=dtype)
        elif isinstance(array, np.ndarray):
            self.array = array
            self.dtype = array.dtype.name
        else:
            raise ValueError('List or np.ndarray required!')

        if self.array.ndim == 1:
            # convert 1-dim array to 2-dim
            self.array = np.reshape(self.array, (-1, 1))

        if self.array.ndim == 3:
            raise ValueError('Dimension of input list >= 3! SolverzArray accepts list with dimension <=2')

    @property
    def row_size(self):
        return self.array.shape[0]

    @property
    def column_size(self):
        return self.array.shape[1]

    def __repr__(self):
        return f"PySAS\n{self.array.__repr__()}"
        # return f"{self.__class__.__name__}: {self.dtype} ({self.row_size}, {self.column_size})"

    def __getitem__(self, item):
        """
        to make PSArry object subscriptable
        :param item:
        :return:
        """
        return self.array.__getitem__(item)

    def __setitem__(self, key, value):
        """
        to make PSArry object support item setting
        :param key:
        :param value:
        :return:
        """
        self.array.__setitem__(key, value)

    def __array__(self):
        return self.array

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method == '__call__':
            if ufunc == np.multiply:
                scalars = []
                for arg in args:
                    if isinstance(arg, Number) or isinstance(arg, np.ndarray):
                        scalars.append(arg)
                    elif isinstance(arg, self.__class__):
                        scalars.append(arg.array)
                    else:
                        raise TypeError(f'Data Type {type(arg)} of {arg} Not Support')
                try:
                    return self.__class__(np.matmul(*scalars, **kwargs))
                except ValueError:
                    return self.__class__(np.multiply(*scalars, **kwargs))
            else:
                # 如果输入含有PSArray，则会导致__array_ufunc__函数的无穷递归，因此需要将PSArray转化为ndarray
                scalars = []
                for arg in args:
                    if isinstance(arg, Number) or isinstance(arg, np.ndarray):
                        scalars.append(arg)
                    elif isinstance(arg, self.__class__):
                        scalars.append(arg.array)
                    else:
                        raise TypeError(f'Data Type {type(arg)} of {arg} Not Support')
                return self.__class__(ufunc(*scalars, **kwargs))
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in Np_Mapping:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle SolverzArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return Np_Mapping[func](*args, **kwargs)


def implements(np_function):
    """Register an __array_function__ implementation for SolverzArray objects."""

    def decorator(func):
        Np_Mapping[np_function] = func
        return func

    return decorator


@implements(np.diag)
def diag(x):
    """
    Generate diagonal matrix of given vector x
    :param x: vector
    :return: diagonal matrix
    """
    if x.column_size == 1 or x.row_size == 1:
        return np.diag(np.reshape(np.asarray(x), (-1,)))
    else:
        raise ValueError(f"Vector input required but {x.column_size} columns detected!")


@implements(np.transpose)
def transpose(x):
    """
    Generate diagonal matrix of given vector x
    :param x: vector
    :return: diagonal matrix
    """
    return np.transpose(np.asarray(x))


def mat_multiply(*args: Union[SolverzArray, np.ndarray, list]):
    """
    Generate multiplication of args
    :param args:
    :return:
    """
    return reduce(lambda x, y: x * y, [SolverzArray(arg) if isinstance(arg, list) else arg for arg in args])


Lambdify_Mapping = {'Mat_Mul': mat_multiply, 'Diagonal': np.diag}

