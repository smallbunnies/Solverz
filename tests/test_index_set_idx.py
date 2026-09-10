"""A plain ``Idx`` of the same name and size as a set index is not treated as set-backed (issue #161)."""
import numpy as np
import sympy as sp

from Solverz import Model, Var, Param, LoopEqn, Set, Idx, Eqn
from Solverz.equation.eqn import SetIdx, _lookup_index_set


def test_set_index_keeps_its_set_and_a_plain_idx_has_none():
    S1 = Set('S1', np.array([4, 0, 2]))
    S2 = Set('S2', np.array([1, 3, 0]))
    i1, i2 = S1.idx('i'), S2.idx('i')
    assert isinstance(i1, SetIdx) and isinstance(i1, sp.Idx)
    assert str(i1) == 'i' and int(i1.lower) == 0 and int(i1.upper) == 2
    assert i1 != i2                                    # same name and size, different sets
    assert _lookup_index_set(i1) is S1 and _lookup_index_set(i2) is S2
    assert _lookup_index_set(i1.func(*i1.args)) is S1  # survives a SymPy rebuild
    assert _lookup_index_set(sp.Idx('i', 3)) is None
    assert _lookup_index_set(Idx('i', 3)) is None
    assert i1 != sp.Idx('i', 3)
    assert S1.idx('i') == i1                           # the same set, the same index


def test_plain_idx_is_not_gathered_through_an_earlier_set():
    a = Model()
    a.x = Var('x', np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
    a.S = Set('S', np.array([4, 0, 2]))
    i = a.S.idx('i')
    a.eqn = LoopEqn('eqn', outer=a.S, body=a.x[i] - 1.0, model=a)
    assert 'S' in a.eqn.var_map                        # the set index is gathered through S

    b = Model()
    b.y = Var('y', np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    b.c = Param('c', np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    k = Idx('i', 3)                                    # same name and size as the set index
    b.eqn = LoopEqn('eqn', outer_index=k, body=b.y[k] - b.c[k], model=b)
    b.tail = Eqn('tail', b.y[3:5] - b.c[3:5])
    assert 'S' not in b.eqn.var_map
    assert not any(str(ib) == 'S' for ib in b.eqn.body.atoms(sp.IndexedBase))
    spf, y0 = b.create_instance()
    np.testing.assert_allclose(spf.g(y0, 'eqn'), [0.9, 1.8, 2.7])
