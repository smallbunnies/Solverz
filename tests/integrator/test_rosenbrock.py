"""The parts of the Rosenbrock step and of the Integrator that the lockstep
of ``test_rosenbrock_lockstep.py`` does not reach.

The built-in methods are the tables of legacy Rodas, and ``from_hairer``
gives the ``rodas4`` arrays bit for bit; a table-only subclass takes the
attempts of legacy Rodas; the two norms of the default configuration and the
legacy norm with its ``1e6`` override; the default configuration takes
legacy's step on an autonomous model, where the two ``dF/dt`` policies
agree; the slopes of the Rodas3 interpolant in both configurations; the
contract of ``interp``; the dispatch of one attempt in both styles, with its
failures and its contract errors; and the residual service.

Every ``rodas.py:N`` refers to legacy Rodas at commit ``056e87a``, before its
deprecation warning moved its lines.
"""
import functools
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_array

from Solverz.integrator import (Algorithm, Rodas3, Rodas4, Rodas5P, Rodasp, Rosenbrock,
                                RosenbrockTableau, StepFailure)
from Solverz.integrator.algorithm import check_style
from Solverz.integrator.integrator import Integrator
from Solverz.integrator.rosenbrock import _pairing
from Solverz.num_api.num_eqn import nDAE
from Solverz.solvers.daesolver.rodas.param import Rodas_param
from Solverz.solvers.option import Opt

from tests.integrator.legacy_rodas import legacy_run
from tests.integrator.test_legacy_transcription import SCHEMES

pytestmark = pytest.mark.i3

TABLE = ('alpha', 'gammatilde', 'a', 'g', 'b', 'bd', 'c', 'd', 'e')

BUILTIN = [(Rodas3, 'rodas3', 'hermite', 2),
           (Rodas4, 'rodas4', 'ntrp1', 3),
           (Rodasp, 'rodasp', 'ntrp1', 3),
           (Rodas5P, 'rodas5p', 'ntrp1', 3)]


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _rodas4_tables(**kwargs):
    """``from_hairer`` on the tables of ``Rodas_param('rodas4')``."""
    ref = Rodas_param('rodas4')
    args = dict(beta=ref.beta, b=ref.b, bd=ref.bd, pord=ref.pord, c=ref.c, d=ref.d, e=ref.e)
    args.update(kwargs)
    return RosenbrockTableau.from_hairer(ref.gamma, ref.alpha.T, **args)


def _replay(integ, a):
    """Put the Integrator in the state of the recorded attempt ``a`` and take it."""
    integ.t, integ.dt = a.t, a.dt
    np.copyto(integ.uprev, a.y0)
    integ.new_step = a.reject == 0
    integ.force_stepfail = False
    integ.perform_step()
    assert not integ.force_stepfail, integ._stepfail_reason


def _accept(integ, t, dt):
    """One attempt from ``uprev`` at ``t``, then the fields the loop sets on
    acceptance, whatever the error."""
    integ.t, integ.dt, integ.new_step = t, dt, True
    integ.perform_step()
    assert not integ.force_stepfail, integ._stepfail_reason
    integ.tprev, integ.dt_step = t, dt
    integ.t = integ.t_step = t + dt
    integ.u_step = integ.u


# -- the methods ----------------------------------------------------------------


@pytest.mark.parametrize('cls, scheme, interpolation, interp_order', BUILTIN,
                         ids=[b[1] for b in BUILTIN])
def test_the_builtin_methods_are_the_legacy_tables(cls, scheme, interpolation, interp_order):
    ref = Rodas_param(scheme)
    tab = cls.tableau
    assert isinstance(tab, Rodas_param)
    for name in TABLE:
        assert _byte_equal(getattr(tab, name), getattr(ref, name)), name
    assert (tab.s, tab.pord, tab.gamma) == (ref.s, ref.pord, ref.gamma)
    assert cls.scheme == scheme
    assert cls.order == cls.error_order == ref.pord
    assert (cls.interpolation, cls.interp_order) == (interpolation, interp_order)
    assert (cls.inplace, cls.adaptive, cls.explicit, cls.norm) == (True, True, False, 'max')
    alg = cls()
    assert alg.tableau is tab and alg.legacy_compat is False
    assert cls(legacy_compat=True).legacy_compat is True
    check_style(alg)


def test_construction_and_from_scheme():
    with pytest.raises(TypeError):
        Rodas4(True)
    with pytest.raises(TypeError):
        # solver=Rodas4 passed without parentheses and then called as legacy Rodas
        Rodas4(None, [0, 1], np.zeros(1), Opt())
    with pytest.raises(TypeError, match='has no tableau'):
        Rosenbrock()
    for cls, scheme, _, _ in BUILTIN:
        alg = Rosenbrock.from_scheme(scheme)
        assert type(alg) is cls and alg.legacy_compat is False
        assert Rosenbrock.from_scheme(scheme, legacy_compat=True).legacy_compat is True
    with pytest.raises(ValueError) as e:
        Rosenbrock.from_scheme('rodas3d')
    assert str(e.value) == ("'rodas3d' defines no dense-output coefficients (param.py:171-200) "
                            "and is not provided by the integrator core")
    for name in ('Rodas4', 'radau', ['rodas4']):
        with pytest.raises(ValueError, match='unknown Rosenbrock scheme'):
            Rosenbrock.from_scheme(name)


def test_from_hairer_gives_the_rodas4_arrays():
    ref = Rodas_param('rodas4')
    for tab in (_rodas4_tables(), _rodas4_tables(beta=None, gamma_ij=ref.beta - ref.alpha.T)):
        for name in TABLE:
            assert _byte_equal(getattr(tab, name), getattr(ref, name)), name
        assert (tab.s, tab.pord, tab.gamma) == (6, 4, 0.25)
        # stored transposed, so that a row of the table is a contiguous column
        assert tab.alpha[:, 3].flags.c_contiguous and tab.gammatilde[:, 3].flags.c_contiguous


def test_from_hairer_refuses_malformed_tables():
    ref = Rodas_param('rodas4')
    alpha, beta = ref.alpha.T, ref.beta
    with pytest.raises(TypeError, match='exactly one of beta and gamma_ij'):
        _rodas4_tables(beta=None)
    with pytest.raises(TypeError, match='exactly one of beta and gamma_ij'):
        _rodas4_tables(gamma_ij=beta - alpha)
    with pytest.raises(ValueError, match='strictly lower triangular'):
        RosenbrockTableau.from_hairer(0.25, alpha + np.eye(6), beta=beta, b=ref.b, bd=ref.bd, pord=4)
    with pytest.raises(ValueError, match='square'):
        RosenbrockTableau.from_hairer(0.25, alpha, beta=beta[:, :5], b=ref.b, bd=ref.bd, pord=4)
    with pytest.raises(ValueError, match='together or not at all'):
        _rodas4_tables(d=None)
    with pytest.raises(ValueError, match='length 6'):
        _rodas4_tables(bd=ref.bd[:5])


def test_a_subclass_takes_its_traits_from_its_tableau():
    dense, plain = _rodas4_tables(), _rodas4_tables(c=None, d=None, e=None)

    class WithDense(Rosenbrock):
        scheme = 'with_dense'
        tableau = dense

    class Plain(Rosenbrock):
        scheme = 'plain'
        tableau = plain

    class Stated(Rosenbrock):
        scheme = 'stated'
        tableau = dense
        interpolation = 'hermite'
        order = 3

    assert (WithDense.interpolation, WithDense.order, WithDense.error_order) == ('ntrp1', 4, 4)
    assert WithDense.interp_order == 1
    assert Plain.interpolation == 'linear'
    assert (Stated.interpolation, Stated.order, Stated.error_order) == ('hermite', 3, 4)
    # a parent's interpolation describes the parent's tableau, so a class that
    # states a tableau of its own and no interpolation derives it anew
    class PlainRodas4(Rodas4):
        tableau = plain

    class DenseRodas3(Rodas3):
        tableau = dense

    class KeptRodas3(Rodas3):
        scheme = 'kept'

    class StatedRodas4(Rodas4):
        tableau = plain
        interp_order = 2

    assert (PlainRodas4.interpolation, DenseRodas3.interpolation) == ('linear', 'ntrp1')
    assert KeptRodas3.interpolation == 'hermite' and KeptRodas3.tableau is Rodas3.tableau
    # and so does the order of the interpolant, which the class may state
    assert (PlainRodas4.interp_order, DenseRodas3.interp_order, KeptRodas3.interp_order) == (1, 1, 2)
    assert StatedRodas4.interp_order == 2
    with pytest.raises(TypeError, match='has no c, d and e'):
        class NoDense(Rosenbrock):
            tableau = plain
            interpolation = 'ntrp1'
    with pytest.raises(TypeError, match='must be one of'):
        class Unknown(Rosenbrock):
            tableau = dense
            interpolation = 'spline'


def test_a_table_only_subclass_takes_the_legacy_attempts(model):
    class TableRodas4(Rosenbrock):
        scheme = 'table_rodas4'
        tableau = _rodas4_tables()

    dae, y0 = model('dae_test')
    kwargs = dict(scheme='rodas4', hinit=0.1)
    attempts = []
    legacy_run(dae, [0, 20], y0.copy(), Opt(**kwargs), attempts)
    integ = Integrator(dae, [0, 20], y0, TableRodas4(legacy_compat=True), Opt(**kwargs))
    for a in attempts:
        _replay(integ, a)
        assert _byte_equal(integ.u, a.ynew) and _byte_equal(integ.EEst, a.err_raw)


# -- the error norms ------------------------------------------------------------


class _RmsRodas4(Rodas4):
    norm = 'rms'


def _norm_case(model, alg, rtol, atol):
    dae, y0 = model('alloc', 'inline_sparse', 5)
    integ = Integrator(dae, [0, 1], y0, alg, Opt(rtol=rtol, atol=atol))
    u, uprev, e = np.random.default_rng(7).standard_normal((3, 5))
    integ.u[:] = u
    integ.uprev[:] = uprev
    return integ, u, uprev, e


ATOLS = [1e-6, np.linspace(1e-8, 1e-6, 5)]


@pytest.mark.parametrize('rtol, atol', [(1e-3, ATOLS[0]), (1e-6, ATOLS[1])], ids=['scalar', 'array'])
def test_the_legacy_norm(model, rtol, atol):
    for alg in (Rodas4(legacy_compat=True), _RmsRodas4(legacy_compat=True)):
        integ, u, _, e = _norm_case(model, alg, rtol, atol)
        got = integ.error_norm(e)
        # rodas.py:208-209, for every algorithm
        ref = np.max(np.abs(e / (atol + rtol * abs(u)).reshape((-1,))))
        assert type(got) is np.float64 and _byte_equal(got, ref)
        for bad in (np.inf, -np.inf, np.nan):
            integ.u[2] = bad
            got = integ.error_norm(e)
            assert type(got) is np.float64 and got == 1.0e6
    # 0/0 with a finite u stays NaN, and the attempt is rejected
    integ, u, _, e = _norm_case(model, Rodas4(legacy_compat=True), rtol, 0.0)
    integ.u[1] = e[1] = 0.0
    with np.errstate(invalid='ignore'):
        assert np.isnan(integ.error_norm(e))


@pytest.mark.parametrize('rtol, atol', [(1e-3, ATOLS[0]), (1e-6, ATOLS[1])], ids=['scalar', 'array'])
def test_the_default_norms(model, rtol, atol):
    for alg, reduce in ((Rodas4(), lambda v: np.max(np.abs(v))),
                        (_RmsRodas4(), lambda v: np.sqrt(np.mean(np.square(v))))):
        integ, u, uprev, e = _norm_case(model, alg, rtol, atol)
        got = integ.error_norm(e)
        ref = reduce(e / (atol + rtol * np.maximum(np.abs(u), np.abs(uprev))))
        assert type(got) is np.float64 and _byte_equal(got, ref)


def test_an_unknown_norm_is_refused(model):
    class L1Rodas4(Rodas4):
        norm = 'l1'

    dae, y0 = model('dae_test')
    with pytest.raises(ValueError, match="must be 'rms' or 'max'"):
        Integrator(dae, [0, 1], y0, L1Rodas4(), Opt())
    # the legacy-compatible configuration uses the legacy norm whatever the algorithm declares
    Integrator(dae, [0, 1], y0, L1Rodas4(legacy_compat=True), Opt())


@pytest.mark.parametrize('scheme', SCHEMES)
def test_the_default_configuration_takes_the_legacy_step_on_an_autonomous_model(model, scheme):
    """``dF/dt`` is exactly zero under both policies when ``F`` does not
    depend on ``t``, so the step is legacy's and only the norm differs."""
    dae, y0 = model('vdp')
    kwargs = dict(scheme=scheme, rtol=1e-6, atol=1e-9)
    attempts = []
    legacy_run(dae, [0, 20], y0.copy(), Opt(**kwargs), attempts)
    integ = Integrator(dae, [0, 20], y0, Rosenbrock.from_scheme(scheme), Opt(**kwargs))
    assert not integ.opts.legacy_compat
    for a in attempts:
        _replay(integ, a)
        assert _byte_equal(integ.u, a.ynew)
        scale = 1e-9 + 1e-6 * np.maximum(np.abs(a.ynew), np.abs(a.y0))
        assert _byte_equal(integ.EEst, np.max(np.abs(integ.cache.utilde / scale)))
        assert not integ.dFdt().any()
    assert any(a.reject > 0 for a in attempts)


# -- interpolation --------------------------------------------------------------


def test_the_pairing_of_rows_and_variables():
    rows, cols, Mv = _pairing(csc_array(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, -1.0]])))
    assert rows.tolist() == [1, 2] and cols.tolist() == [0, 2] and Mv.tolist() == [2.0, -1.0]
    # an explicit zero, as ModeSwitch keeps on a pinned row, pairs nothing
    M = csc_array((np.array([1.0, 0.0]), np.array([0, 1]), np.array([0, 1, 2])), shape=(2, 2))
    assert M.nnz == 2
    rows, cols, Mv = _pairing(M)
    assert rows.tolist() == [0] and cols.tolist() == [0] and Mv.tolist() == [1.0]
    rows, cols, Mv = _pairing(np.array([[0.0, 3.0], [0.0, 0.0]]))
    assert rows.tolist() == [0] and cols.tolist() == [1] and Mv.tolist() == [3.0]
    assert _pairing(csc_array(np.array([[1.0, 1.0], [0.0, 0.0]]))) is None
    assert _pairing(csc_array(np.array([[1.0, 0.0], [1.0, 0.0]]))) is None


def test_the_legacy_rodas3_slopes_are_the_residuals(model):
    dae, y0 = model('permuted')
    integ = Integrator(dae, [0, 1], y0, Rodas3(legacy_compat=True), Opt())
    _accept(integ, 0.0, 0.05)
    n = integ.stats.nfeval
    integ.interp(0.02)
    integ.interp(0.03)
    # one end residual per interpolated step; F at the start is the step's own
    assert integ.stats.nfeval == n + 1
    c = integ.cache
    assert _byte_equal(c.s0, dae.F(0.0, integ.uprev, dae.p))
    assert _byte_equal(c.s1, dae.F(integ.t_step, integ.u_step, dae.p))


def test_the_default_rodas3_slopes_pair_rows_and_variables(model):
    dae, y0 = model('permuted')
    integ = Integrator(dae, [0, 1], y0, Rodas3(), Opt())
    c = integ.cache
    _accept(integ, 0.0, 0.05)
    integ.interp(0.02)
    rows, cols, Mv = c.pairing
    r, k = dae.M.nonzero()
    assert _byte_equal(rows, r) and _byte_equal(cols, k) and rows.size == 1
    assert rows[0] != cols[0], 'the rows of M are aligned with the variables'
    alg_var = 1 - cols[0]
    F0 = dae.F(0.0, integ.uprev, dae.p)
    F1 = dae.F(integ.t_step, integ.u_step, dae.p)
    secant = (integ.u_step - integ.uprev) / integ.dt_step
    assert c.s0[cols[0]] == F0[rows[0]] / Mv[0] and c.s1[cols[0]] == F1[rows[0]] / Mv[0]
    assert c.s0[alg_var] == c.s1[alg_var] == secant[alg_var]
    for theta in (0.25, 0.5, 0.75):
        y = integ.interp(theta * 0.05)
        lin = integ.uprev + theta * (integ.u_step - integ.uprev)
        assert abs(y[alg_var] - lin[alg_var]) <= 1e-12 * (1 + abs(lin[alg_var]))
    # the pairing is kept while the model is unchanged and rebuilt when it changes
    pairing = c.pairing
    np.copyto(integ.uprev, integ.u)
    _accept(integ, 0.05, 0.05)
    integ.interp(0.07)
    assert c.pairing is pairing
    integ.model_epoch += 1
    _accept(integ, 0.1, 0.05)
    integ.interp(0.12)
    assert c.pairing is not pairing and c.pairing_epoch == integ.model_epoch


def test_the_default_rodas3_slopes_divide_by_the_entries_of_M():
    """``2 x' = -x`` in row 0, ``-0.5 w' = 0.5 w`` in row 1 and ``0 = z - x/2``
    in row 2, with ``y = (z, x, w)``: the slope of a paired variable is its
    residual row divided by its entry of ``M``, which is not 1."""
    M = csc_array(np.array([[0.0, 2.0, 0.0], [0.0, 0.0, -0.5], [0.0, 0.0, 0.0]]))

    def F(t, y, p, out=None):
        out = np.empty(3) if out is None else out
        out[0] = -y[1]
        out[1] = 0.5 * y[2]
        out[2] = y[0] - 0.5 * y[1]
        return out

    def J(t, y, p):
        return csc_array(np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.5], [1.0, -0.5, 0.0]]))

    integ = Integrator(nDAE(M, F, J, {}), [0, 1], np.array([0.5, 1.0, 1.0]), Rodas3(), Opt())
    _accept(integ, 0.0, 0.1)
    integ.interp(0.04)
    c = integ.cache
    rows, cols, Mv = c.pairing
    assert rows.tolist() == [0, 1] and cols.tolist() == [1, 2] and Mv.tolist() == [2.0, -0.5]
    F0 = F(0.0, integ.uprev, None)
    F1 = F(integ.t_step, integ.u_step, None)
    assert _byte_equal(c.s0[cols], F0[rows] / Mv) and _byte_equal(c.s1[cols], F1[rows] / Mv)
    # the derivatives of the model itself
    assert c.s0[1] == -integ.uprev[1] / 2.0 and c.s0[2] == -integ.uprev[2]
    secant = (integ.u_step - integ.uprev) / integ.dt_step
    assert c.s0[0] == c.s1[0] == secant[0]


def test_the_default_rodas3_slopes_without_a_pairing_are_secants():
    # d(x + z)/dt = -x - z, 0 = z - x/2: one row of M holds two variables
    M = csc_array(np.array([[1.0, 1.0], [0.0, 0.0]]))

    def F(t, y, p, out=None):
        out = np.empty(2) if out is None else out
        out[0] = -y[0] - y[1]
        out[1] = y[1] - 0.5 * y[0]
        return out

    def J(t, y, p):
        return csc_array(np.array([[-1.0, -1.0], [-0.5, 1.0]]))

    integ = Integrator(nDAE(M, F, J, {}), [0, 1], np.array([1.0, 0.5]), Rodas3(), Opt())
    _accept(integ, 0.0, 0.1)
    y = integ.interp(0.04)
    c = integ.cache
    assert c.pairing is None
    secant = (integ.u_step - integ.uprev) / integ.dt_step
    assert _byte_equal(c.s0, secant) and _byte_equal(c.s1, secant)
    lin = integ.uprev + 0.4 * (integ.u_step - integ.uprev)
    assert np.all(np.abs(y - lin) <= 1e-12 * (1 + np.abs(lin)))


@pytest.mark.parametrize('cls', [Rodas3, Rodas4])
def test_interp(model, cls):
    dae, y0 = model('dae_test')
    integ = Integrator(dae, [0, 1], y0, cls(legacy_compat=True), Opt())
    _accept(integ, 0.0, 0.1)
    start = integ.interp(0.0)
    assert _byte_equal(start, integ.uprev) and start is not integ.uprev
    end = integ.interp(integ.t)
    assert _byte_equal(end, integ.u) and end is not integ.u
    out = np.full(2, np.nan)
    assert integ.interp(0.05, out) is out
    ref = np.full(2, np.nan)
    integ.alg.interpolant(integ, integ.cache, 0.05 / 0.1, ref)
    assert _byte_equal(out, ref)
    for tq in (-1e-3, integ.t + 1e-9):
        with pytest.raises(ValueError, match='outside the last step'):
            integ.interp(tq)
    # after the state changed outside a step only its own time is accepted
    integ._interp_valid = False
    assert _byte_equal(integ.interp(integ.t), integ.u)
    with pytest.raises(ValueError, match='only tq == t'):
        integ.interp(0.05)


# -- one attempt ----------------------------------------------------------------


def _linear(F, J, M=None):
    return nDAE(csc_array(np.eye(2)) if M is None else M, F, J, {})


def _F_decay(t, y, p, out=None):
    out = np.empty(2) if out is None else out
    np.negative(y, out=out)
    return out


def _J_decay(t, y, p):
    return csc_array(-np.eye(2))


def test_the_services_of_the_start_of_a_step(model):
    dae, y0 = model('trace')
    integ = Integrator(dae, [0, 1], y0, Rodas4(legacy_compat=True), Opt())
    integ.t, integ.dt = 0.3, 0.01
    integ.perform_step()
    F0, ft = integ.F0(), integ.dFdt()
    assert not F0.flags.writeable and not ft.flags.writeable
    assert _byte_equal(F0, dae.F(0.3, integ.uprev, dae.p))
    out = np.empty(y0.size)
    assert integ.dFdt(out=out) is out and _byte_equal(out, ft)
    # a retry keeps them, and after acceptance they still describe the start
    n = (integ.stats.nfeval, integ.stats.nJeval)
    J0 = integ.J0()
    integ.dt = 0.005
    integ.perform_step()
    assert integ.J0() is J0 and _byte_equal(integ.F0(), F0)
    assert (integ.stats.nfeval, integ.stats.nJeval) == (n[0] + Rodas4.tableau.s - 1, n[1])
    integ.tprev, integ.t = 0.3, 0.305
    assert _byte_equal(integ.F0(), F0) and integ.stats.nfeval == n[0] + Rodas4.tableau.s - 1


def test_an_arithmetic_error_in_F_fails_the_attempt():
    def F(t, y, p, out=None):
        if t > 0:
            raise ZeroDivisionError('float division by zero')
        return _F_decay(t, y, p, out)

    integ = Integrator(_linear(F, _J_decay), [0, 1], np.ones(2), Rodas4(legacy_compat=True), Opt())
    integ.t, integ.dt = 0.0, 0.1
    integ.perform_step()
    assert integ.force_stepfail and integ.new_step is False
    assert integ._stepfail_reason == 'ZeroDivisionError in F: float division by zero'


def test_a_singular_iteration_matrix_fails_the_attempt(backend):
    def J(t, y, p):
        return csc_array((2, 2))

    M = csc_array(np.ones((2, 2)))
    integ = Integrator(_linear(_F_decay, J, M), [0, 1], np.ones(2), Rodas4(legacy_compat=True), Opt())
    integ.t, integ.dt = 0.0, 0.1
    integ.perform_step()
    assert integ.force_stepfail
    assert integ._stepfail_reason.startswith('the factorization of W failed')
    # a zero row of W divides by zero in the row scaling, which raises under
    # np.seterr(all='raise') and fails the attempt as well
    M = csc_array(np.array([[1.0, 0.0], [0.0, 0.0]]))
    # from a consistent start, since DaeIc cannot solve 0 = -y[1] with J = 0
    integ = Integrator(_linear(_F_decay, J, M), [0, 1], np.array([1.0, 0.0]), Rodas4(legacy_compat=True),
                       Opt())
    assert not integ.failed
    integ.t, integ.dt = 0.0, 0.1
    with np.errstate(all='raise'):
        integ.perform_step()
    assert integ.force_stepfail and integ._stepfail_reason.startswith('FloatingPointError')


class _NoEstimate(Algorithm):
    scheme = 'no_estimate'
    order = 1
    error_order = 1
    adaptive = True
    inplace = True

    def perform_step(self, integ, cache):
        np.copyto(integ.u, integ.uprev)


def test_an_in_place_adaptive_algorithm_sets_its_error(model):
    dae, y0 = model('dae_test')
    integ = Integrator(dae, [0, 1], y0, _NoEstimate(), Opt())
    integ.dt = 0.1
    with pytest.raises(TypeError, match='no_estimate.perform_step set no error estimate'):
        integ.perform_step()
    integ = Integrator(dae, [0, 1], y0, _NoEstimate(), Opt(fix_h=True, hinit=0.1))
    integ.dt = 0.1
    integ.perform_step()
    assert integ.EEst is None and not integ.force_stepfail


class _Formula(Algorithm):
    scheme = 'formula'
    order = 1
    error_order = 1
    adaptive = True

    def __init__(self, result):
        self.result = result

    def perform_step(self, s):
        return self.result(s)


class _FixedFormula(_Formula):
    scheme = 'fixed_formula'
    adaptive = False


def test_the_formula_dispatch(model):
    dae, y0 = model('dae_test')
    alg = _Formula(lambda s: (2.0 * s.y0, np.full(s.n, 1e-3)))
    integ = Integrator(dae, [0, 1], y0, alg, Opt(rtol=1e-3, atol=1e-6))
    integ.dt = 0.1
    integ.perform_step()
    assert _byte_equal(integ.u, 2.0 * integ.uprev)
    scale = 1e-6 + 1e-3 * np.maximum(np.abs(integ.u), np.abs(integ.uprev))
    assert _byte_equal(integ.EEst, np.sqrt(np.mean(np.square(np.full(2, 1e-3) / scale))))
    for result, message in ((lambda s: 2.0 * s.y0, 'returned no error estimate'),
                            (lambda s: (np.zeros(3), None), 'returned y of shape'),
                            (lambda s: np.zeros((2, 1)), 'returned y of shape'),
                            (lambda s: (s.y0, np.zeros(3)), 'error estimate of shape')):
        alg.result = result
        with pytest.raises(TypeError, match=message):
            integ.perform_step()
    # with a fixed step the estimate of an adaptive method is not read
    integ = Integrator(dae, [0, 1], y0, _Formula(lambda s: (s.y0, s.y0)), Opt(fix_h=True, hinit=0.1))
    integ.dt = 0.1
    integ.perform_step()
    assert integ.EEst is None
    integ = Integrator(dae, [0, 1], y0, _FixedFormula(lambda s: (s.y0, s.y0)), Opt(hinit=0.1))
    integ.dt = 0.1
    with pytest.raises(TypeError, match='returned an error estimate but declares adaptive = False'):
        integ.perform_step()
    alg = _FixedFormula(lambda s: s.y0 + s.h)
    integ = Integrator(dae, [0, 1], y0, alg, Opt(hinit=0.1))
    integ.dt = 0.1
    integ.perform_step()
    assert _byte_equal(integ.u, integ.uprev + 0.1) and integ.EEst is None


class _NoStep(Algorithm):
    # adaptive, so that without the style check the construction would succeed
    scheme = 'no_step'
    order = 1
    adaptive = True


class _InplaceOfOneParameter(_NoStep):
    scheme = 'inplace_of_one'
    inplace = True

    def perform_step(self, s):
        return s.y0


def test_the_integrator_checks_the_style_before_anything_else(model):
    dae, y0 = model('dae_test')
    for alg, message in ((Algorithm(), 'Algorithm does not define perform_step'),
                         (_NoStep(), '_NoStep does not define perform_step'),
                         (_InplaceOfOneParameter(), 'takes 1 parameters after self, but inplace = True needs 2')):
        with pytest.raises(TypeError, match=message):
            Integrator(dae, [0, 1], y0, alg, Opt())


# -- the residual service -------------------------------------------------------


def _F_plain(t, y, p):
    return np.array([y[1] - t, -y[0]])


def test_the_residual_service():
    y = np.array([0.5, -2.0])
    M = csc_array(np.eye(2))

    def F_out(t, y, p, out=None):
        out = np.empty(2) if out is None else out
        out[:] = _F_plain(t, y, p)
        return out

    @functools.wraps(F_out)
    def wrapped(*args):
        return F_out(*args)

    def build(F):
        return Integrator(SimpleNamespace(M=M, F=F, J=_J_decay, p={}), [0, 1], y, Rodas4(), Opt())

    assert build(F_out)._F is F_out
    # the adapter of nDAE is recognized by its own signature and not wrapped again
    dae = nDAE(M, _F_plain, _J_decay, {})
    assert dae.F is not _F_plain and build(dae.F)._F is dae.F
    for F in (_F_plain, wrapped):
        integ = build(F)
        # DaeIc evaluates the residual once at initialization
        assert integ.stats.nfeval == 1
        r1, r2 = integ.F(0.25, y), integ.F(0.25, y)
        assert r1 is not r2 and _byte_equal(r1, _F_plain(0.25, y, {}))
        out = np.full(2, np.nan)
        assert integ.F(0.25, y, out=out) is out and _byte_equal(out, r1)
        assert integ.stats.nfeval == 4
    integ = build(F_out)
    assert integ.stats.nJeval == 0
    assert integ.J(0.0, y) is not None and integ.stats.nJeval == 1


def test_arithmetic_errors_of_the_model_become_step_failures():
    def F(t, y, p, out=None):
        raise ZeroDivisionError('boom')

    def J(t, y, p):
        raise OverflowError('big')

    def F_value(t, y, p, out=None):
        # from t > 0 on, so that DaeIc at t = 0 passes
        if t > 0:
            raise ValueError('not a failure')
        return _F_decay(t, y, p, out)

    y = np.ones(2)
    # DaeIc at initialization already meets the error and fails the run
    integ = Integrator(_linear(F, J), [0, 1], y, Rodas4(), Opt())
    assert integ.failed
    with pytest.raises(StepFailure, match='^ZeroDivisionError in F: boom$'):
        integ.F(0.0, y)
    with pytest.raises(StepFailure, match='^OverflowError in J: big$'):
        integ.J(0.0, y)
    integ = Integrator(_linear(F_value, _J_decay), [0, 1], y, Rodas4(), Opt())
    with pytest.raises(ValueError, match='not a failure'):
        integ.F(0.5, y)
    # nor does the dispatch catch it
    integ.t, integ.dt = 0.0, 0.1
    with pytest.raises(ValueError, match='not a failure'):
        integ.perform_step()
