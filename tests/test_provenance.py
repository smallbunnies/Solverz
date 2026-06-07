# tests/test_provenance.py
import numpy as np
from Solverz import Eqn, Ode, Var, Model, Idx, LoopEqn, Sum, Param


def test_equations_have_source_default_none():
    e = Eqn('e', Var('x', 1) - 1)
    o = Ode('o', Var('y', 1), Var('y', 1))
    assert e.source is None
    assert o.source is None  # Ode inherits Eqn.__init__


def test_loopeqn_has_source_default_none():
    m = Model()
    m.x = Var('x', np.ones(3))
    m.b = Param('b', np.zeros(3))
    i = Idx('i', 3)
    m.le = LoopEqn('le', outer_index=i, body=m.x[i] - m.b[i], model=m)
    assert m.le.source is None
