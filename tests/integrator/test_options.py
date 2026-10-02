"""``IntegratorOptions.from_opt`` reads ``Opt`` once and never writes to it.

Every ``rodas.py:N`` refers to legacy Rodas at commit ``056e87a``, before its
deprecation warning moved its lines.
"""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from Solverz.integrator import IntegratorOptions
from Solverz.solvers.laesolver import linsolver, resolve_backend
from Solverz.solvers.option import Opt


def _alg(adaptive=True, legacy_compat=False, scheme='toy'):
    return SimpleNamespace(adaptive=adaptive, legacy_compat=legacy_compat, scheme=scheme)


def _snapshot(opt):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(opt).items()}


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], np.ndarray):
            assert np.array_equal(a[k], b[k]), k
        else:
            assert a[k] is b[k] or a[k] == b[k], k


def test_fields_map_from_opt():
    event = lambda t, y: (y, [1], [0])
    opt = Opt(rtol=1e-5, atol=1e-7, f_savety=0.8, fac1=0.3, fac2=5, facmax=4,
              hinit=0.01, hmax=0.5, event=event, pbar=True, linsolver='superlu')
    o = IntegratorOptions.from_opt(opt, _alg(), [0.0, 2.0])
    assert (o.rtol, o.atol) == (1e-5, 1e-7)
    assert (o.safety, o.qmin, o.qmax, o.qmax_init) == (0.8, 0.3, 5, 4)
    assert (o.dt0, o.dtmax) == (0.01, 0.5)
    assert o.event is event
    assert o.pbar is True
    assert o.linsolver == 'superlu'
    assert o.adaptive is True
    assert o.legacy_compat is False
    assert (o.t0, o.tend) == (0.0, 2.0)
    assert o.dense is False and o.saveat is None
    assert (o.failfactor, o.max_consecutive_reject) == (2.0, 100)


def test_opt_is_never_written():
    opt = Opt(atol=np.array([1e-8, 1e-6]))
    before = _snapshot(opt)
    for alg in (_alg(), _alg(legacy_compat=True)):
        IntegratorOptions.from_opt(opt, alg, [0, 20])
        IntegratorOptions.from_opt(opt, alg, np.linspace(0, 1, 11))
        _same(before, _snapshot(opt))
    assert opt.hmax is None


def test_opt_none_means_default_opt():
    o = IntegratorOptions.from_opt(None, _alg(), [0, 1])
    d = Opt()
    assert (o.rtol, o.atol, o.safety, o.qmin, o.qmax, o.qmax_init) == \
        (d.rtol, d.atol, d.f_savety, d.fac1, d.fac2, d.facmax)
    assert o.dt0 is None


def test_options_are_frozen():
    o = IntegratorOptions.from_opt(Opt(), _alg(), [0, 1])
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.rtol = 1.0


def test_default_configuration_converts_times_to_float():
    o = IntegratorOptions.from_opt(Opt(hinit=1), _alg(), [0, 20])
    for x in (o.t0, o.tend, o.dt0, o.dtmax):
        assert type(x) is float
    assert (o.t0, o.tend, o.dt0, o.dtmax) == (0.0, 20.0, 1.0, 20.0)


def test_legacy_configuration_keeps_the_types_of_legacy():
    # rodas.py:71 keeps the dtype of tspan and rodas.py:77 takes |tend - t0| in it.
    o = IntegratorOptions.from_opt(Opt(hinit=1), _alg(legacy_compat=True), [0, 20])
    assert type(o.t0) is np.int64 and type(o.tend) is np.int64
    assert type(o.dtmax) is np.int64 and o.dtmax == 20
    assert type(o.dt0) is int
    o = IntegratorOptions.from_opt(Opt(), _alg(legacy_compat=True), np.arange(5))
    assert o.saveat.dtype == np.arange(5).dtype


def test_dtmax():
    assert IntegratorOptions.from_opt(Opt(), _alg(), [0.5, 2.0]).dtmax == 1.5
    assert IntegratorOptions.from_opt(Opt(hmax=0.1), _alg(), [0.5, 2.0]).dtmax == 0.1


def test_saveat_is_a_read_only_copy_of_the_nodes():
    tspan = np.linspace(0, 1, 11)
    o = IntegratorOptions.from_opt(Opt(), _alg(), tspan)
    assert o.dense
    assert o.saveat.dtype == np.float64
    assert o.saveat.tobytes() == tspan[1:].tobytes()
    assert not o.saveat.flags.writeable
    tspan[5] = 7.0
    assert o.saveat[4] == 0.5


def test_array_tolerances_are_read_only_copies():
    atol = np.array([1e-8, 1e-6])
    o = IntegratorOptions.from_opt(Opt(atol=atol), _alg(), [0, 1])
    assert o.atol is not atol and not o.atol.flags.writeable
    assert o.atol.tobytes() == atol.tobytes()
    atol[0] = 1.0
    assert o.atol[0] == 1e-8
    o = IntegratorOptions.from_opt(Opt(atol=[1, 2]), _alg(), [0, 1])
    assert o.atol.dtype == np.float64
    # a 0-d array is an array too, and keeps its dtype
    rtol = np.array(1e-6, dtype=np.float32)
    o = IntegratorOptions.from_opt(Opt(rtol=rtol), _alg(), [0, 1])
    assert o.rtol is not rtol and not o.rtol.flags.writeable
    assert o.rtol.dtype == np.float32 and o.rtol.tobytes() == rtol.tobytes()
    rtol[()] = 1.0
    assert o.rtol == np.float32(1e-6)


def test_adaptive():
    assert IntegratorOptions.from_opt(Opt(), _alg(), [0, 1]).adaptive is True
    assert IntegratorOptions.from_opt(Opt(fix_h=True, hinit=0.1), _alg(), [0, 1]).adaptive is False
    assert IntegratorOptions.from_opt(Opt(hinit=0.1), _alg(adaptive=False), [0, 1]).adaptive is False


def test_linsolver_is_resolved_when_read():
    with linsolver('superlu'):
        o = IntegratorOptions.from_opt(Opt(), _alg(), [0, 1])
    assert o.linsolver == 'superlu'
    with linsolver('klu'):
        o = IntegratorOptions.from_opt(Opt(), _alg(), [0, 1])
    # 'superlu' when libklu is missing, as every solver of Solverz degrades it
    assert o.linsolver == resolve_backend('klu')


@pytest.mark.parametrize('name', ['klu', 'superlu'])
def test_a_linsolver_of_the_call_overrides_the_global_one(name):
    other = 'superlu' if name == 'klu' else 'klu'
    with linsolver(other):
        o = IntegratorOptions.from_opt(Opt(linsolver=name), _alg(), [0, 1])
    assert o.linsolver == resolve_backend(name)


def test_t0_equal_to_tend_is_accepted():
    o = IntegratorOptions.from_opt(Opt(), _alg(), [1.0, 1.0])
    assert o.t0 == o.tend == 1.0 and o.dtmax == 0.0


@pytest.mark.parametrize('legacy', [False, True])
def test_t0_after_tend_raises_the_legacy_text(legacy):
    with pytest.raises(ValueError, match=r'^t0: 1\.0 > tend: 0\.5$'):
        IntegratorOptions.from_opt(Opt(), _alg(legacy_compat=legacy), [1.0, 0.5])


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('hinit', [0, 0.0, -1e-3])
def test_hinit_not_positive_raises(legacy, hinit):
    with pytest.raises(ValueError, match='opt.hinit'):
        IntegratorOptions.from_opt(Opt(hinit=hinit), _alg(legacy_compat=legacy), [0, 1])


def test_non_increasing_grid_raises_in_the_default_configuration_only():
    for tspan in ([0, 1, 1, 2], [0, 2, 1, 3]):
        with pytest.raises(ValueError, match='increase strictly'):
            IntegratorOptions.from_opt(Opt(), _alg(), tspan)
        IntegratorOptions.from_opt(Opt(), _alg(legacy_compat=True), tspan)


def test_fixed_step_without_hinit_raises():
    with pytest.raises(ValueError, match=r'^opt\.fix_h needs opt\.hinit$'):
        IntegratorOptions.from_opt(Opt(fix_h=True), _alg(), [0, 1])
    with pytest.raises(ValueError, match=r'^toy has no error estimate \(adaptive = False\)'):
        IntegratorOptions.from_opt(Opt(), _alg(adaptive=False), [0, 1])
    with pytest.raises(ValueError, match=r'^toy has no error estimate'):
        IntegratorOptions.from_opt(Opt(fix_h=True), _alg(adaptive=False), [0, 1])
