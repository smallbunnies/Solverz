"""Regression tests for Model.add() name-collision detection (#130)."""
import pytest

from Solverz import Eqn, Model, Param, Var


def test_add_distinct_names():
    m1 = Model()
    m1.a = Param('a', 1.0)
    m2 = Model()
    m2.b = Param('b', 2.0)
    m1.add(m2)
    assert m1.a.v == 1.0
    assert m1.b.v == 2.0


def test_add_collision_differing_values_raises():
    m1 = Model()
    m1.Cp = Param('Cp', 4186)
    m2 = Model()
    m2.Cp = Param('Cp', 4200)
    with pytest.raises(ValueError, match="Cp"):
        m1.add(m2)


def test_add_collision_equal_values_allowed():
    m1 = Model()
    m1.Cp = Param('Cp', 4186)
    m2 = Model()
    m2.Cp = Param('Cp', 4186)
    m1.add(m2)
    assert m1.Cp.v == 4186


def test_add_collision_shared_object_allowed():
    shared = Param('Cp', 4186)
    m1 = Model()
    m1.Cp = shared
    m2 = Model()
    m2.Cp = shared
    m1.add(m2)
    assert m1.Cp is shared


def test_add_single_attribute_collision_detected():
    m = Model()
    m.Cp = Param('Cp', 4186)
    with pytest.raises(ValueError, match="Cp"):
        m.add(Param('Cp', 4200))


def test_add_dict_with_collision():
    m = Model()
    m.a = Param('a', 1.0)
    with pytest.raises(ValueError, match="'a'"):
        m.add({'a': Param('a', 2.0)})
