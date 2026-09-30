"""Import hygiene of ``Solverz.integrator``.

Importing any ``Solverz.solvers`` submodule first runs
``Solverz/solvers/__init__.py``, which imports every legacy solver, so a
legacy module that imported the integrator would close a cycle. The
integrator therefore imports at module level only the Solverz modules in
``ALLOWED``, and no module outside it imports the integrator;
``Solverz/__init__.py`` is the one exception, since it imports the solvers
before it re-exports names of the integrator.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

import Solverz
import Solverz.integrator

PACKAGE = Path(Solverz.__file__).parent
INTEGRATOR = PACKAGE / 'integrator'

ALLOWED = {
    'Solverz.solvers.option',
    'Solverz.solvers.stats',
    'Solverz.solvers.solution',
    'Solverz.solvers.parser',
    'Solverz.solvers.laesolver',
    'Solverz.solvers.klu_backend',
    'Solverz.solvers.daesolver.daeic',
    'Solverz.solvers.daesolver.rodas.param',
    'Solverz.solvers.daesolver.rodas.rodas',
    'Solverz.num_api.num_eqn',
    'Solverz.variable.variables',
}


def _module_level_imports(path):
    """Absolute names imported by the statements that run at import time,
    including those under a module-level ``if`` or ``try``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names = []
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = path.parent.relative_to(PACKAGE.parent).parts
                base = '.'.join(pkg[:len(pkg) - node.level + 1])
                names.append(f"{base}.{node.module}" if node.module else base)
            else:
                names.append(node.module)
        elif isinstance(node, (ast.If, ast.Try)):
            for field in ('body', 'orelse', 'finalbody'):
                stack += getattr(node, field, [])
            for handler in getattr(node, 'handlers', []):
                stack += handler.body
    return names


def _all_imports(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_integrator_imports_only_the_allowed_solverz_modules():
    files = sorted(INTEGRATOR.glob('*.py'))
    assert files
    for path in files:
        for name in _module_level_imports(path):
            if name == 'Solverz' or name.startswith('Solverz.'):
                assert name in ALLOWED or name.startswith('Solverz.integrator'), \
                    f"{path.name} imports {name} at module level"


def test_no_module_outside_the_integrator_imports_it():
    for path in sorted(PACKAGE.rglob('*.py')):
        rel = path.relative_to(PACKAGE)
        if rel.parts[0] == 'integrator' or rel == Path('__init__.py'):
            continue
        if {'test', 'tests'} & set(rel.parts[:-1]):
            continue
        for name in _all_imports(path):
            assert not name.startswith('Solverz.integrator'), f"{rel} imports {name}"


@pytest.mark.parametrize('module', ['Solverz.integrator'] + sorted(
    f"Solverz.integrator.{p.stem}" for p in INTEGRATOR.glob('*.py') if p.stem != '__init__'))
def test_each_module_imports_first_in_a_fresh_interpreter(module):
    """A cycle shows only when the module is the first one imported."""
    subprocess.run([sys.executable, '-c', f'import {module}'], check=True)


def test_star_import_binds_exactly_all():
    ns = {}
    exec('from Solverz.integrator import *', ns)
    del ns['__builtins__']
    assert sorted(ns) == sorted(Solverz.integrator.__all__)
    assert len(set(Solverz.integrator.__all__)) == len(Solverz.integrator.__all__)


def test_star_import_of_solverz_binds_no_solve_or_init():
    ns = {}
    exec('from sympy import *\nfrom Solverz import *', ns)
    assert 'init' not in ns
    import sympy
    assert ns['solve'] is sympy.solve
    ns = {}
    exec('from Solverz import *', ns)
    assert 'solve' not in ns
    assert 'init' not in ns
