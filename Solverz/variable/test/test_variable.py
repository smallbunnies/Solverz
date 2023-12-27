import numpy as np

from Solverz.variable.variables import as_Vars, combine_Vars, Vars, TimeVars
from Solverz.symboli_algebra.symbols import Var


def test_Vars():
    # as_Vars
    a = Var('a', [1, 2, 3])
    b = Var('b', [4, 5, 6])
    y = as_Vars([a, b])
    assert np.all(np.isclose(y.array, [1, 2, 3, 4, 5, 6]))

    c = Var('c', [0, -2, 3.0])
    d = Var('d', [100, 5, 6])
    z = as_Vars([c, d])
    assert np.all(np.isclose(z.array, [0, -2, 3.0, 100, 5, 6]))

    # combine_Vars
    x = combine_Vars(y, z)
    assert np.all(np.isclose(x.array, [1, 2, 3, 4, 5, 6, 0, -2, 3.0, 100, 5, 6]))

    # get_item
    assert np.all(np.isclose(y['a'], [1.0, 2.0, 3.0]))

    # set_item
    y['a'] = [7, 8, 9]

    try:
        y['a'] = [1]
    except ValueError as e:
        assert e.args[0] == 'Incompatible input array shape!'

    try:
        y['e'] = [1]
    except ValueError as e:
        assert e.args[0] == 'There is no variable e!'

    assert y.__repr__() == "Variables (size 6) ['a', 'b']"

    # operator
    xr = np.array([1, 2, 3, 4, 5, 6, 0, -2, 3.0, 100, 5, 6])
    # Mul
    assert np.all(np.isclose((x * 2).array, 2 * xr))
    assert np.all(np.isclose((x * 2)['c'], [0, -4, 6.0]))
    assert np.all(np.isclose((2 * x).array, 2 * xr))
    assert np.all(np.isclose((2 * x)['c'], [0, -4, 6.0]))

    try:
        np.all(np.isclose((y * z)['c'], [0, -4, 6.0]))
    except ValueError as v:
        assert v.args[0] == 'Cannot multiply Vars object by another one with different variables!'
    assert np.all(np.isclose((y * y)['a'], [49., 64., 81.]))

    # Add
    assert np.all(np.isclose((2 + x).array, 2 + xr))
    assert np.all(np.isclose((2 + x)['c'], [2, 0, 5]))
    assert np.all(np.isclose((x + 2).array, 2 + xr))
    assert np.all(np.isclose((x + 2)['c'], [2, 0, 5]))

    # Sub
    assert np.all(np.isclose((2 - x).array, 2 - xr))
    assert np.all(np.isclose((2 - x)['c'], [2, 4, -1]))
    assert np.all(np.isclose((x - 2).array, xr - 2))
    assert np.all(np.isclose((x - 2)['c'], [-2, -4, 1]))

    # Div
    assert np.all(np.isclose((x / 2).array, xr / 2))
    assert np.all(np.isclose((2 / y).array, 2 / np.array([7, 8, 9, 4, 5, 6])))
    assert np.all(np.isclose((x / 2)['d'], [50., 2.5, 3.]))
    assert np.all(np.isclose((2 / y)['b'], [0.5, 0.4, 0.33333333]))

    # derive_alias
    x0 = x.derive_alias('0')
    assert np.all(np.isclose(x0.array, x.array))
    assert np.all(np.isclose(x0['a0'], x['a']))


def test_TimeVars():
    a = Var('a', [1, 2, 3])
    b = Var('b', [4, 5, 6])
    y = as_Vars([a, b])
    yt = TimeVars(y, length=1001)
    assert yt.a == y.a

    # __getitem__()
    assert np.all(np.isclose(yt[0].array, [1., 2., 3., 4., 5., 6.]))
    assert np.all(np.isclose(yt['a'], yt.array[:, 0:3]))
    assert np.all(np.isclose(yt[0:5].array, yt.array[0:5, :]))

    # __setitem__()
    yt[2] = 2 * y
    assert np.all(np.isclose(yt[2].array, [2., 4., 6., 8., 10., 12.]))
    assert yt.len == 1001

    # T
    assert np.all(np.isclose(yt.T, yt.array.T))

    # __repr__()
    assert yt.__repr__() == "Time-series (size 1001×6) ['a', 'b']"
