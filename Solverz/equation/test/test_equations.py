import numpy as np
import pytest

from Solverz import Eqn, Model, Var


def test_create_instance_sequences_and_partial_jacobian():
    m = Model()
    m.x = Var('x', [1.0, 2.0])
    m.y = Var('y', [3.0])
    m.eqn1 = Eqn('eqn1', m.x - 1)
    m.eqn2 = Eqn('eqn2', m.y - m.x[0])

    eqs, y0 = m.create_instance(eqn_sequence=['eqn2', 'eqn1'],
                                var_sequence=['y', 'x'])

    assert eqs.a.object_list == ['eqn2', 'eqn1']
    assert y0.a.object_list == ['y', 'x']
    np.testing.assert_allclose(y0.array, np.array([3.0, 1.0, 2.0]))

    partial_jac = eqs.FormPartialJac(y0, ['eqn1'], ['x'])
    np.testing.assert_array_equal(partial_jac.shape, np.array([2, 2]))
    np.testing.assert_array_equal(partial_jac.coordinate0, np.array([1, 1]))

    row, col, data = partial_jac.parse_row_col_data()
    np.testing.assert_array_equal(row, np.array([1, 2]))
    np.testing.assert_array_equal(col, np.array([1, 2]))
    np.testing.assert_allclose(data, np.array([1.0, 1.0]))


def test_form_partial_jacobian_requires_contiguous_lists():
    m = Model()
    m.x = Var('x', [1.0])
    m.y = Var('y', [2.0])
    m.eqn1 = Eqn('eqn1', m.x - 1)
    m.eqn2 = Eqn('eqn2', m.y - m.x)

    eqs, y0 = m.create_instance(eqn_sequence=['eqn2', 'eqn1'],
                                var_sequence=['y', 'x'])

    with pytest.raises(ValueError, match="discontinuous"):
        eqs.FormPartialJac(y0, ['eqn1', 'eqn2'], ['y', 'x'])
