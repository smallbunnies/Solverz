from sympy import Number as symNumber
import numpy as np
from numbers import Number


def is_number(num):
    if isinstance(num, (Number, symNumber)):
        return True
    elif isinstance(num, np.ndarray) and num.size == 1:
        return True
    else:
        return False
