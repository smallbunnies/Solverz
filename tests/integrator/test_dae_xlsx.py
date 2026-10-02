"""The integrator against the reference sheets of ``tests/dae_test.xlsx``.

I4: ``Rodas4(legacy_compat=True)`` reproduces the sheets ``rodas``, the nine
accepted steps of legacy Rodas on ``[0, 20]``, and ``rodas_dense``, its 201
nodes, within ``1e-8``, as ``tests/test_dae.py`` checks legacy Rodas.

I5: ``Rodas4()`` in the default configuration takes other steps, so it is
compared with ``rodas_dense`` only, within ``1e-2``; the sheet ``rodas`` has
no time column and holds legacy's own steps.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Solverz import made_numerical
from Solverz.integrator import Rodas4
from Solverz.solvers.option import Opt

from tests.integrator import models

DATA = Path(__file__).resolve().parent.parent / 'dae_test.xlsx'


@pytest.fixture(scope='module')
def sheets():
    return pd.read_excel(DATA, sheet_name=None, engine='openpyxl')


@pytest.fixture(scope='module')
def dae_test():
    sdae, y0 = models.dae_test_model()
    return made_numerical(sdae, y0, sparse=True), y0


@pytest.mark.i4
@pytest.mark.parametrize('sheet, tspan', [('rodas', [0, 20]), ('rodas_dense', np.linspace(0, 20, 201))],
                         ids=['rodas', 'rodas_dense'])
def test_the_compatible_rodas4_reproduces_the_sheets(sheets, dae_test, sheet, tspan):
    dae, y0 = dae_test
    sol = Rodas4(legacy_compat=True)(dae, tspan, y0, Opt(hinit=0.1))
    ref = np.asarray(sheets[sheet])
    assert sol.stats.ret == 'success'
    assert ref.shape == sol.Y.array.shape
    assert np.max(np.abs(ref - sol.Y.array)) < 1e-8


@pytest.mark.i5
def test_the_default_rodas4_is_close_to_the_dense_sheet(sheets, dae_test):
    dae, y0 = dae_test
    sol = Rodas4()(dae, np.linspace(0, 20, 201), y0, Opt(hinit=0.1))
    ref = np.asarray(sheets['rodas_dense'])
    assert sol.stats.ret == 'success'
    assert ref.shape == sol.Y.array.shape
    dev = np.max(np.abs(ref - sol.Y.array))
    print(f"default Rodas4 against the sheet rodas_dense: max deviation {dev:.3e}")
    assert dev < 1e-2
