import numpy as np
from sympy import lambdify
import inspect

from Solverz.sym_algebra.symbols import iVar, Para, idx, IdxVar, IdxPara, Idxidx
from Solverz.num_api.custom_function import sol_slice


# test of IndexPrinter
def test_IndexPrinter():
    x = iVar('x')
    G = Para('G', dim=2)
    k = idx('k')
    M = idx('M')
    j = idx('j')
    assert x[0].__repr__() == 'x[0]'
    assert x[k].__repr__() == 'x[k]'
    assert x[k ** 2].__repr__() == 'x[k**2]'
    assert x[k ** 2].SymInIndex.__repr__() == "{'k': k}"
    assert x[[1, k, M[j]]].__repr__() == 'x[[1, k, M[j]]]'
    assert x[[1, k, M[j]]].SymInIndex.__repr__() == "{'k': k, 'M': M, 'j': j}"
    assert x[[1, 2, 3]].__repr__() == 'x[[1, 2, 3]]'
    assert x[1:k - 1:2].__repr__() == 'x[1:k - 1:2]'
    assert x[1:k - 1:2].SymInIndex.__repr__() == "{'k': k}"
    assert x[M[j]:k[j] - 1].__repr__() == 'x[M[j]:k[j] - 1]'
    assert x[M[j]:k[j] - 1].SymInIndex.__repr__() == "{'M': M, 'j': j, 'k': k}"
    assert G[k, :].__repr__() == 'G[k,:]'
    assert G[k, :].SymInIndex.__repr__() == "{'k': k}"
    assert G[:, k[M[j]]].__repr__() == 'G[:,k[M[j]]]'
    assert G[:, k[M[j]]].SymInIndex.__repr__() == "{'k': k, 'M': M, 'j': j}"
    assert G[[1, 2, 3], k].__repr__() == 'G[[1, 2, 3],k]'


modules = [{'sol_slice': sol_slice}, 'numpy']


# test of numpy code printer
def test_numpy_code_printer():
    x = iVar('x')
    k = idx('k')
    fxk = lambdify([x, k], x[k], modules=modules)
    assert (inspect.getsource(fxk) ==
            "def _lambdifygenerated(x, k):\n    return x[k]\n")
    x = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    k = np.array([0, 2, 5])
    assert np.isclose(fxk(x, k), np.array([[1], [3], [6]])).all()

    z = iVar('z')
    fz = lambdify([z], z[1], modules=modules)
    assert (inspect.getsource(fz) ==
            "def _lambdifygenerated(z):\n    return z[1]\n")
    x = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    assert fz(x) == np.array([2])

    x = iVar('x')
    yy = iVar('yy')
    k = idx('k')
    fxyk = lambdify([x, yy, k], x[[1, 2, 3]] + yy[k], modules=modules)
    assert (inspect.getsource(fxyk) ==
            "def _lambdifygenerated(x, yy, k):\n    return x[[1, 2, 3]] + yy[k]\n")
    x = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    yy = np.array([-1, -2, -3, -7]).reshape((-1, 1))
    k = np.array([0, 3, 3])
    assert np.isclose(fxyk(x, yy, k), np.array([[1], [-4], [-3]])).all()

    z = iVar('z')
    k = idx('k')
    fzk = lambdify([z, k], z[1:k], modules=modules)
    assert (inspect.getsource(fzk) ==
            "def _lambdifygenerated(z, k):\n    return z[sol_slice(1, k, None)]\n")
    z = np.array([-1, -2, -3, -7]).reshape((-1, 1))
    k = np.array([3])
    assert np.isclose(fzk(z, k), np.array([[-2], [-3]])).all()

    z = iVar('z')
    M = idx('M')
    j = idx('j')
    fzMj = lambdify([z, M, j], z[1:M[j] + 2], modules=modules)
    assert (inspect.getsource(fzMj) ==
            "def _lambdifygenerated(z, M, j):\n    return z[sol_slice(1, M[j] + 2, None)]\n")
    z = np.array([-1, -2, -3, -7]).reshape((-1, 1))
    M = np.array([0, 3, 1])
    j = np.array([1])
    assert np.isclose(fzMj(z, M, j), np.array([[-2], [-3], [-7]])).all()

    k = idx('k')
    G = Para('G', dim=2)
    fGk = lambdify([k, G], G[k, :], modules=modules)
    assert (inspect.getsource(fGk) ==
            "def _lambdifygenerated(k, G):\n    return G[k,:]\n")
    G = np.random.randn(3, 3)
    k = np.array([1, 2])
    assert np.isclose(fGk(k, G), G[k, :]).all()

    k = idx('k')
    G = Para('G', dim=2)
    j = idx('j')
    fkjG = lambdify([k, j, G], G[:, 1:k[j] - 1:2], modules=modules)
    assert (inspect.getsource(fkjG) ==
            "def _lambdifygenerated(k, j, G):\n    return G[:,sol_slice(1, k[j] - 1, 2)]\n")
    G = np.random.randn(3, 3)
    k = np.array([0, 2])
    j = [1]
    assert np.isclose(fkjG(k, j, G), G[:, 1:k[j][0] - 1:2]).all()
