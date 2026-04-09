import numpy as np

from Solverz import AE, Eqn, atan2, made_numerical
from Solverz.sym_algebra.symbols import iVar
from Solverz.variable.variables import as_Vars


def test_atan2_made_numerical():
    x = iVar('x', [3.0, 4.0])
    f = Eqn('f', atan2(x[1], x[0]))

    ae = AE([f])
    y = as_Vars([x])
    n_ae, code = made_numerical(ae, y, sparse=True, output_code=True)

    np.testing.assert_allclose(
        n_ae.F(y, n_ae.p),
        np.array([np.arctan2(4.0, 3.0)]),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        n_ae.J(y, n_ae.p).toarray(),
        np.array([[-4.0 / 25.0, 3.0 / 25.0]]),
        rtol=1e-10,
        atol=1e-10,
    )
    assert 'atan2(' in code['F']
    assert 'atan2(' in code['J'] or 'atan2(' in code['F']
