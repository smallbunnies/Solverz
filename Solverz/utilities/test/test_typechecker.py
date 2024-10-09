import numpy as np
from sympy import Integer, Float
from Solverz.utilities.type_checker import is_integer, is_zero


def test_is_number():
    assert is_integer(np.array([1.0]).astype(int)[0])
    assert not is_integer(np.array([1.0])[0])

def test_is_zero():
    assert is_zero(Integer(0))
    assert not is_zero(Float(1))
    assert is_zero(Float(0))
    assert is_zero(0)
    assert is_zero(0.)
    assert is_zero(np.array(0))
