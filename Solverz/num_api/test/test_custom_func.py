import numpy as np

from Solverz.num_api.custom_function import In, LessThan, GreaterThan, And, Or, Not, minmod, minmod_flag, switch_minmod


def test_in():
    a = np.array([1, 2, 3.0])
    b = np.array([3, 2, 1.0])
    np.testing.assert_allclose(In(a, 2, 2), np.array([0, 1, 0]))
    np.testing.assert_allclose(In(a, 1, 3), np.array([1, 1, 1]))
    np.testing.assert_allclose(In(a, 1, 2.5), np.array([1, 1, 0]))
    np.testing.assert_allclose(In(5, 4, 4.5), np.array([0]))


def test_gt():
    a = np.array([1, 2, 3.0])
    b = np.array([3, 2, 1.0])
    np.testing.assert_allclose(GreaterThan(a, b), np.array([0, 0, 1]))
    np.testing.assert_allclose(GreaterThan(5, 4), np.array([1]))
    np.testing.assert_allclose(GreaterThan(5, 6), np.array([0]))


def test_lt():
    a = np.array([1, 2, 3.0])
    b = np.array([3, 2, 1.0])
    np.testing.assert_allclose(LessThan(a, b), np.array([1, 0, 0]))
    np.testing.assert_allclose(LessThan(5, 4), np.array([0]))
    np.testing.assert_allclose(LessThan(5, 6), np.array([1]))


def test_and():
    a = np.array([1, 0, 1])
    b = np.array([1, 1, 0])
    np.testing.assert_allclose(And(a, b), np.array([1, 0, 0]))
    np.testing.assert_allclose(And(1, 0), np.array([0]))
    np.testing.assert_allclose(And(1, 1), np.array([1]))


def test_or():
    a = np.array([1, 0, 1])
    b = np.array([1, 1, 0])
    np.testing.assert_allclose(Or(a, b), np.array([1, 1, 1]))
    np.testing.assert_allclose(Or(1, 0), np.array([1]))
    np.testing.assert_allclose(Or(0, 0), np.array([0]))


def test_not():
    a = np.array([1, 0, 1])
    np.testing.assert_allclose(Not(a), np.array([0, 1, 0]))
    np.testing.assert_allclose(Not(1), np.array([0]))
    np.testing.assert_allclose(Not(0), np.array([1]))

def test_minmod():
    a = np.array([-1, 1, 2, 4.])
    b = np.array([-3, 2, 0, 0])
    c = np.array([-5, 3, -1, 1.])
    np.testing.assert_allclose(minmod(a, b, c), np.array([-1, 1, 0, 0.]))

def test_minmod_flag():
    a = np.array([-1, 3, 2, -4,   5, 10])
    b = np.array([-3, 2, 1, -2,   0, -1])
    c = np.array([-5, 1, 1, -5.4, 1, 2])
    np.testing.assert_allclose(minmod_flag(a, b, c), np.array([1, 3, 2, 2, 0, 0]))

def test_switch_minmod():
    a = np.array([-1, 3, 2, -4,   5, 10])
    b = np.array([-3, 2, 1, -2,   0, -1])
    c = np.array([-5, 1, 1, -5.4, 1, 2])
    flag = minmod_flag(a, b, c)
    np.testing.assert_allclose(switch_minmod(a, b, c, flag), np.array([-1, 1, 1, -2, 0, 0.0]))

