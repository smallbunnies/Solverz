import numpy as np
import pytest

from Solverz.utilities.miscellaneous import derive_incidence_matrix, rearrange_list


def test_derive_incidence_matrix():
    incidence = derive_incidence_matrix([2, 2, 0], 3, 4)

    np.testing.assert_array_equal(incidence.toarray(),
                                  np.array([[0, 0, 1, 0],
                                            [0, 0, 1, 0],
                                            [1, 0, 0, 0]]))


def test_rearrange_list():
    assert rearrange_list(['eqn1', 'eqn2', 'eqn3'], ['eqn3', 'eqn1']) == ['eqn3', 'eqn1', 'eqn2']

    with pytest.raises(ValueError, match="duplicate"):
        rearrange_list(['eqn1', 'eqn2'], ['eqn1', 'eqn1'])

    with pytest.raises(ValueError, match="not found"):
        rearrange_list(['eqn1', 'eqn2'], ['eqn3'])
