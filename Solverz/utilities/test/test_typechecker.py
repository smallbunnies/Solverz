import numpy as np
from Solverz.utilities.type_checker import is_integer


def test_is_number():
    assert is_integer(np.array([1.0]).astype(int)[0])
    assert not is_integer(np.array([1.0])[0])
