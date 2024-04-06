import numpy as np

from Solverz.num_api.custom_function import SolIn, SolLessThan, SolGreaterThan


def test_in():
    a = np.array([1, 2, 3.0])
    b = np.array([3, 2, 1.0])
    np.testing.assert_allclose(SolIn(a, 2, 2), np.array([0, 1, 0]))
    np.testing.assert_allclose(SolIn(a, 1, 3), np.array([1, 1, 1]))
    np.testing.assert_allclose(SolIn(a, 1, 2.5), np.array([1, 1, 0]))
    np.testing.assert_allclose(SolIn(5, 4, 4.5), np.array([0]))


def test_gt():
    a = np.array([1, 2, 3.0])
    b = np.array([3, 2, 1.0])
    np.testing.assert_allclose(SolGreaterThan(a, b), np.array([0, 0, 1]))
    np.testing.assert_allclose(SolGreaterThan(5, 4), np.array([1]))
    np.testing.assert_allclose(SolGreaterThan(5, 6), np.array([0]))


def test_lt():
    a = np.array([1, 2, 3.0])
    b = np.array([3, 2, 1.0])
    np.testing.assert_allclose(SolLessThan(a, b), np.array([1, 0, 0]))
    np.testing.assert_allclose(SolLessThan(5, 4), np.array([0]))
    np.testing.assert_allclose(SolLessThan(5, 6), np.array([1]))
