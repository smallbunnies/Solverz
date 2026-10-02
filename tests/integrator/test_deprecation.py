"""The deprecation of legacy Rodas.

Legacy ``Rodas`` warns with a ``DeprecationWarning`` that points at its caller,
whether it is called through the ``dae_io_parser`` wrapper or as
``Rodas.__wrapped__``, so that Python's default filters show it in a script
and hide it elsewhere. The integrator's algorithms do not warn. The filter of
``pyproject.toml`` hides the warning in the test suite, which calls legacy
Rodas on purpose, and must match the start of the message.
"""
import re
import sys
import warnings
from pathlib import Path

import pytest

from Solverz import Rodas as TopLevelRodas
from Solverz.integrator import Rodas4, solve
from Solverz.solvers.daesolver.rodas.rodas import Rodas
from Solverz.solvers.option import Opt

from tests.integrator import models

pytestmark = pytest.mark.i8

MESSAGE = ("Rodas is deprecated. Use Rodas3, Rodas4, Rodasp or Rodas5P from Solverz.integrator, "
           "called as Rodas4()(dae, tspan, y0, opt); the class selects the method, not opt.scheme. "
           "Rodas4(legacy_compat=True) and the other three reproduce Rodas on adaptive runs without "
           "events, given the class that matches opt.scheme and an Opt that no earlier Rodas call has "
           "changed; the user guide lists the conditions.")

PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'


def _here():
    """The line after the caller's current one."""
    return sys._getframe(1).f_lineno + 1


def _deprecations(record):
    """The deprecation warnings of legacy Rodas in ``record``; a deprecation
    that a newer NumPy, SciPy or tqdm emits on the same path is not one."""
    return [w for w in record if issubclass(w.category, DeprecationWarning)
            and str(w.message).startswith('Rodas is deprecated')]


def _assert_points_here(record, line):
    found = _deprecations(record)
    assert len(found) == 1, [str(w.message) for w in found]
    w = found[0]
    assert w.category is DeprecationWarning
    assert str(w.message).startswith('Rodas is deprecated')
    assert str(w.message) == MESSAGE
    assert Path(w.filename).resolve() == Path(__file__).resolve()
    assert w.lineno == line


def test_rodas_through_its_wrapper_warns_once_at_the_callers_line():
    dae, y0 = models.build('dae_test')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        line = _here()
        sol = Rodas(dae, [0, 1], y0, Opt())
    _assert_points_here(record, line)
    assert sol.T[-1] == 1


def test_rodas_of_the_top_level_namespace_warns_at_the_callers_line():
    assert TopLevelRodas is Rodas
    dae, y0 = models.build('dae_test')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        line = _here()
        TopLevelRodas(dae, [0, 1], y0)
    _assert_points_here(record, line)


def test_rodas_called_unwrapped_warns_once_at_the_callers_line():
    dae, y0 = models.build('dae_test')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        line = _here()
        sol = Rodas.__wrapped__(dae, [0, 1], y0, Opt())
    _assert_points_here(record, line)
    assert sol.T[-1] == 1


def test_every_call_warns():
    dae, y0 = models.build('dae_test')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        lines = []
        for _ in range(3):
            lines.append(_here())
            Rodas(dae, [0, 1], y0.copy())
    found = _deprecations(record)
    assert [w.lineno for w in found] == lines


def _python_default_filters():
    """The two filters of Python's defaults that concern ``DeprecationWarning``,
    in their order: shown once per location in ``__main__``, ignored elsewhere."""
    warnings.resetwarnings()
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    # inserted in front, so it is tried first, as in warnings.filters
    warnings.filterwarnings('default', category=DeprecationWarning, module='__main__')


def test_the_default_filters_show_the_warning_in_a_script_only():
    """Python shows a ``DeprecationWarning`` by default only when it is
    attributed to ``__main__``, which the stack level makes of a script."""
    dae, y0 = models.build('dae_test')
    with warnings.catch_warnings(record=True) as record:
        _python_default_filters()
        Rodas(dae, [0, 1], y0)
    assert _deprecations(record) == []

    namespace = {'Rodas': Rodas, 'dae': dae, 'y0': y0, '__name__': '__main__'}
    source = compile('Rodas(dae, [0, 1], y0)\n', '<script>', 'exec')
    with warnings.catch_warnings(record=True) as record:
        _python_default_filters()
        exec(source, namespace)
    found = _deprecations(record)
    assert len(found) == 1
    assert found[0].filename == '<script>' and found[0].lineno == 1


def test_the_integrator_algorithms_do_not_warn():
    dae, y0 = models.build('dae_test')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        Rodas4()(dae, [0, 1], y0, Opt())
        Rodas4(legacy_compat=True)(dae, [0, 1], y0, Opt())
        solve(dae, [0, 1], y0)
    assert _deprecations(record) == []


def test_the_pyproject_filter_matches_the_message():
    text = PYPROJECT.read_text(encoding='utf-8')
    entries = re.findall(r'^filterwarnings\s*=\s*\[(.*)\]\s*$', text, flags=re.MULTILINE)
    assert len(entries) == 1
    filters = re.findall(r'"([^"]*)"', entries[0])
    assert 'ignore:Rodas is deprecated:DeprecationWarning' in filters
    action, message, _ = 'ignore:Rodas is deprecated:DeprecationWarning'.split(':')

    dae, y0 = models.build('dae_test')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        warnings.filterwarnings(action, message=message, category=DeprecationWarning)
        Rodas(dae, [0, 1], y0)
    assert _deprecations(record) == []
