# Provenance Docstrings in Generated Modules — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every `module_printer(...).render()` artifact self-describing — the module docstring states the generating Solverz version and a generation timestamp plus a provenance/equation table, and every generated equation function (F residuals and J derivative kernels) carries a docstring naming its equation (and, when known, its source component + package@version).

**Architecture:** Solverz core gains (1) an optional `source` dict on `Eqn`/`LoopEqn`, (2) a `stamp_source(model, …)` helper that walks a model fragment and tags its equations, and (3) printer changes that read `getattr(eqn, 'source', None)` and emit docstrings. SolMuseum and SolPSDyn each call `stamp_source` once per component, reading their own `__version__`. Solverz never imports the downstream packages; the version travels as data on the equation objects. Plain (unstamped) models degrade gracefully to name-only docstrings.

**Tech Stack:** Python 3.11, Solverz (sympy-based code printer, Numba module path), pytest. Repos: `Solverz-dev` (core), `SolMuseum`, `SolPSDyn` — all editable installs in the `sim` conda env (`/opt/miniconda3/envs/sim/bin/python`).

**Spec:** `docs/superpowers/specs/2026-06-06-solverz-module-provenance-design.md`.

**Test command (Solverz):** `/opt/miniconda3/envs/sim/bin/python -m pytest tests/ Solverz/ -x -q`
**Run one test:** `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -v`

---

## File Structure

**Solverz-dev (Part A):**
- Modify `Solverz/equation/eqn.py` — add `self.source = None` to `Eqn.__init__` and `LoopEqn.__init__`.
- Create `Solverz/equation/source.py` — `stamp_source()` + `format_source()` helpers (one focused responsibility: provenance stamping/formatting). Keeps `eqn.py` from growing.
- Modify `Solverz/__init__.py` — export `stamp_source`.
- Modify `Solverz/code_printer/python/module/module_printer.py` — F-function docstrings (`print_sub_inner_F`), J-function + loop-jac-kernel docstrings (`print_inner_J`), and a shared `_with_docstring()` string helper.
- Modify `Solverz/code_printer/python/module/module_generator.py` — build the manifest + `{eqn_name: source}` map in `render_modules`, thread them to `print_inner_J` and `print_init_code`; rewrite the module docstring in `print_init_code`.
- Create `tests/test_provenance.py` — all Part A tests.
- Modify `docs/src/release_notes.md` — release note.

**SolMuseum (Part B):**
- Modify each component module under `SolMuseum/SolMuseum/ae/` and `SolMuseum/SolMuseum/dae/` — one `stamp_source(...)` call per `.mdl()`.
- Create `SolMuseum/SolMuseum/<tests path>/test_provenance.py` — provenance test.

**SolPSDyn (Part C):**
- Modify each component under `SolPSDyn/SolPSDyn/dae/` and the wiring in `SolPSDyn/SolPSDyn/system.py`.
- Create a provenance test under SolPSDyn's test dir.

---

# PART A — Solverz core

### Task A1: `source` attribute on equations

**Files:**
- Modify: `Solverz/equation/eqn.py` (`Eqn.__init__` near line 43; `LoopEqn.__init__` near line 1104)
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -v`
Expected: FAIL with `AttributeError: 'Eqn' object has no attribute 'source'`.

- [ ] **Step 3: Add `self.source = None` to `Eqn.__init__`**

In `Solverz/equation/eqn.py`, in `Eqn.__init__`, after `self.derivatives: Dict[str, EqnDiff] = dict()` (line ~43) add:

```python
        # Optional provenance: None, or a dict
        # {'component': str, 'package': str, 'version': str} set by
        # stamp_source(). Read defensively elsewhere via
        # getattr(eqn, 'source', None) so old pickles keep working.
        self.source = None
```

- [ ] **Step 4: Add `self.source = None` to `LoopEqn.__init__`**

`LoopEqn.__init__` bypasses `Eqn.__init__`, so add the attribute explicitly. In `Solverz/equation/eqn.py`, in `LoopEqn.__init__`, right after `self.var_map = dict(var_map)` (line ~1104) add:

```python
        self.source = None  # see Eqn.source; LoopEqn bypasses Eqn.__init__
```

- [ ] **Step 5: Run to verify it passes**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add Solverz/equation/eqn.py tests/test_provenance.py
git commit -m "feat(eqn): add optional source provenance attribute to Eqn/LoopEqn"
```

---

### Task A2: `stamp_source` + `format_source` helpers

**Files:**
- Create: `Solverz/equation/source.py`
- Modify: `Solverz/__init__.py` (after line 14, `from Solverz.model.basic import Model`)
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_provenance.py
from Solverz import stamp_source


def test_stamp_source_tags_all_equations():
    m = Model()
    m.x = Var('x', np.ones(2))
    m.b = Param('b', np.zeros(2))
    m.e0 = Eqn('e0', m.x[0] - m.b[0])
    m.e1 = Eqn('e1', m.x[1] - m.b[1])
    stamp_source(m, component='demo', package='Pkg', version='1.2.3')
    assert m.e0.source == {'component': 'demo', 'package': 'Pkg', 'version': '1.2.3'}
    assert m.e1.source == {'component': 'demo', 'package': 'Pkg', 'version': '1.2.3'}


def test_stamp_source_overwrite_false_preserves_existing():
    m = Model()
    m.x = Var('x', np.ones(1))
    m.e = Eqn('e', m.x[0])
    m.e.source = {'component': 'inner', 'package': 'Sub', 'version': '0.1'}
    stamp_source(m, component='outer', package='Pkg', version='9.9')
    assert m.e.source['component'] == 'inner'  # not clobbered
    stamp_source(m, component='outer', package='Pkg', version='9.9', overwrite=True)
    assert m.e.source['component'] == 'outer'


def test_stamp_source_ignores_non_equations():
    m = Model()
    m.x = Var('x', np.ones(1))     # a Var, not an equation
    m.b = Param('b', np.zeros(1))  # a Param
    m.e = Eqn('e', m.x[0])
    stamp_source(m, component='c', package='P', version='1')
    # Vars/Params untouched; only the Eqn is stamped.
    assert m.e.source is not None
    assert not hasattr(m.x, 'source') or m.x.source is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -k stamp -v`
Expected: FAIL with `ImportError: cannot import name 'stamp_source'`.

- [ ] **Step 3: Create `Solverz/equation/source.py`**

```python
"""Provenance stamping for Solverz equations.

A model fragment assembled by a reusable component (e.g. a SolMuseum or
SolPSDyn building block) can tag every equation it contributes with a
``source`` dict ``{'component', 'package', 'version'}``. The module
printer renders that provenance into the generated module's docstrings.
Solverz core never imports the downstream packages; the version flows in
as data via this helper.
"""
from __future__ import annotations

from Solverz.equation.eqn import Eqn


def stamp_source(model, *, component, package, version, overwrite=False):
    """Set ``.source`` on every equation attribute of ``model``.

    Walks the model's attributes (where freshly built ``Eqn`` / ``Ode`` /
    ``LoopEqn`` / ``LoopOde`` objects live before ``create_instance``) and
    stamps each with ``{'component', 'package', 'version'}``.

    Parameters
    ----------
    model : Solverz.Model
    component, package, version : str
        Provenance fields. ``component`` is the element/model type (e.g.
        ``'gt'``, ``'eps_network'``), ``package`` the distribution (e.g.
        ``'SolMuseum'``), ``version`` its version string.
    overwrite : bool, default False
        When False, equations already carrying a (more specific) source
        are left untouched, so a composite block does not clobber the
        stamps of sub-fragments it aggregates.

    Returns
    -------
    model : the same object, for chaining.
    """
    src = {'component': str(component),
           'package': str(package),
           'version': str(version)}
    for value in vars(model).values():
        if isinstance(value, Eqn):  # covers Ode/LoopEqn/LoopOde (all subclass Eqn)
            if overwrite or getattr(value, 'source', None) is None:
                value.source = dict(src)
    return model


def format_source(source):
    """Return a one-line ``' — <package> <version> / <component>'`` suffix
    for a source dict, or ``''`` when ``source`` is falsy."""
    if not source:
        return ''
    return (f" — {source.get('package', '?')} "
            f"{source.get('version', '?')} / {source.get('component', '?')}")
```

- [ ] **Step 4: Export from `Solverz/__init__.py`**

After `from Solverz.model.basic import Model` (line 14) add:

```python
from Solverz.equation.source import stamp_source, format_source
```

- [ ] **Step 5: Run to verify it passes**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -k stamp -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add Solverz/equation/source.py Solverz/__init__.py tests/test_provenance.py
git commit -m "feat(eqn): add stamp_source/format_source provenance helpers"
```

---

### Task A3: docstring-insertion string helper

**Files:**
- Modify: `Solverz/code_printer/python/module/module_printer.py` (add helper near top, after imports)
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_provenance.py
from Solverz.code_printer.python.module.module_printer import _with_docstring


def test_with_docstring_inserts_after_def_line():
    src = "def inner_F0(x, y):\n    return x + y\n"
    out = _with_docstring(src, 'P_eqn — Pkg 1.0 / comp')
    lines = out.split('\n')
    assert lines[0] == 'def inner_F0(x, y):'
    assert lines[1].strip().startswith('"""')
    assert 'P_eqn' in lines[1]
    assert lines[2].strip() == 'return x + y'


def test_with_docstring_sanitizes_to_single_line():
    src = "def f():\n    return 1\n"
    out = _with_docstring(src, 'line1\nline2  with   spaces')
    doc_line = out.split('\n')[1]
    assert doc_line.count('"""') == 2          # opening + closing on one line
    assert 'line1 line2 with spaces' in doc_line


def test_with_docstring_noop_when_empty_doc():
    src = "def f():\n    return 1\n"
    assert _with_docstring(src, '') == src
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -k with_docstring -v`
Expected: FAIL with `ImportError: cannot import name '_with_docstring'`.

- [ ] **Step 3: Add the helper**

In `Solverz/code_printer/python/module/module_printer.py`, after the import block at the top of the file, add:

```python
def _with_docstring(func_src: str, doc: str) -> str:
    """Insert ``doc`` as a one-line docstring immediately after the
    ``def …:`` line of a rendered function-source string.

    ``func_src`` is the text of a single function whose first ``def``
    line ends with ``:``. ``doc`` is sanitised to one physical line
    (collapsed whitespace, no triple quotes). Returns ``func_src``
    unchanged when ``doc`` is empty. Works uniformly for the AST-rendered
    (pycode) F/J functions and the string-rendered LoopEqn / loop-jac
    kernels because all of them start with a ``def`` line.
    """
    if not doc:
        return func_src
    doc = ' '.join(doc.split()).replace('"""', "'''")
    lines = func_src.split('\n')
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if stripped.startswith('def ') and ln.rstrip().endswith(':'):
            indent = ' ' * (len(ln) - len(stripped) + 4)
            lines.insert(i + 1, f'{indent}"""{doc}"""')
            break
    return '\n'.join(lines)
```

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -k with_docstring -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add Solverz/code_printer/python/module/module_printer.py tests/test_provenance.py
git commit -m "feat(printer): add _with_docstring helper for generated functions"
```

---

### Task A4: F-function docstrings in `print_sub_inner_F`

**Files:**
- Modify: `Solverz/code_printer/python/module/module_printer.py` (`print_sub_inner_F`, the three `code_blocks.append(...)` sites: LoopEqn branch ~line 1290, fast path ~line 1306, Mat_Mul path ~line 1369)
- Test: `tests/test_provenance.py`

The three append sites all have `eqn` (the Eqn object, with `.source`) and `eqn_name` in scope. Wrap each appended string in `_with_docstring`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_provenance.py
import os, sys, tempfile, importlib, uuid


def _render(model, jit=False):
    """create_instance + module_printer render; return the module dir."""
    from Solverz import module_printer
    spf, y0 = model.create_instance()
    d = tempfile.mkdtemp(prefix='sz_prov_')
    name = f'prov_{uuid.uuid4().hex[:8]}'
    module_printer(spf, y0, name, directory=d, jit=jit).render()
    return os.path.join(d, name)


def _read(path, fname):
    with open(os.path.join(path, fname)) as f:
        return f.read()


def _demo_model_with_source():
    m = Model()
    m.x = Var('x', np.ones(2))
    m.b = Param('b', np.array([1.0, 2.0]))
    m.e0 = Eqn('e0', m.x[0] - m.b[0])
    m.e1 = Eqn('e1', m.x[1] - m.b[1])
    stamp_source(m, component='demo', package='Pkg', version='1.2.3')
    return m


def test_f_functions_have_source_docstrings():
    mod = _render(_demo_model_with_source(), jit=False)
    num_func = _read(mod, 'num_func.py')
    # Each inner_F has a docstring naming its equation + source.
    assert '"""e0' in num_func or '"""e1' in num_func
    assert 'Pkg 1.2.3 / demo' in num_func


def test_f_functions_name_only_when_unsourced():
    m = Model()
    m.x = Var('x', np.ones(1))
    m.b = Param('b', np.array([1.0]))
    m.e = Eqn('e', m.x[0] - m.b[0])
    mod = _render(m, jit=False)
    num_func = _read(mod, 'num_func.py')
    assert '"""e"""' in num_func          # name-only docstring
    assert ' / ' not in num_func.split('"""e"""')[0][-200:]  # no source suffix nearby
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -k f_functions -v`
Expected: FAIL (no docstrings emitted yet).

- [ ] **Step 3: Add the docstring import + wrap the three appends**

At the top of `print_sub_inner_F` (or module scope), ensure access to `format_source`:

```python
from Solverz.equation.source import format_source
```

LoopEqn branch — change:
```python
        code_blocks.append(eqn.print_njit_source(f'inner_F{count}'))
```
to:
```python
        _doc = f"{eqn_name}{format_source(getattr(eqn, 'source', None))}"
        code_blocks.append(_with_docstring(
            eqn.print_njit_source(f'inner_F{count}'), _doc))
```

Fast path — change:
```python
        code_blocks.append(pycode(fd, fully_qualified_modules=False))
```
to:
```python
        _doc = f"{eqn_name}{format_source(getattr(eqn, 'source', None))}"
        code_blocks.append(_with_docstring(
            pycode(fd, fully_qualified_modules=False), _doc))
```

Mat_Mul path — apply the identical wrap to its `code_blocks.append(pycode(fd, fully_qualified_modules=False))`.

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -k f_functions -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the existing printer tests (no regression)**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest Solverz/code_printer -q`
Expected: PASS (docstrings don't change semantics).

- [ ] **Step 6: Commit**

```bash
git add Solverz/code_printer/python/module/module_printer.py tests/test_provenance.py
git commit -m "feat(printer): docstring each F equation function with name + source"
```

---

### Task A5: J-kernel docstrings in `print_inner_J`

**Files:**
- Modify: `Solverz/code_printer/python/module/module_printer.py` (`print_inner_J`: add a `source_map=None` parameter; annotate the scalar `inner_J{count}` append ~line 426 and the loop-jac kernel append ~line ~310)
- Modify: `Solverz/code_printer/python/module/module_generator.py` (pass `source_map` to `print_inner_J`, line 98)
- Test: `tests/test_provenance.py`

`print_inner_J(var_address, PARAM, jac, nstep)` does not currently receive equation sources. Add an optional `source_map` (`{eqn_name: source_dict}`). In its block loop, `eqn_name` and `var` are in scope at both append sites.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_provenance.py
def test_j_kernels_have_derivative_docstrings():
    mod = _render(_demo_model_with_source(), jit=False)
    num_func = _read(mod, 'num_func.py')
    # Scalar J kernels annotated as d(eqn)/d(var) with source.
    assert 'd(e0)/d(x)' in num_func or 'd(e1)/d(x)' in num_func
    assert 'Pkg 1.2.3 / demo' in num_func.split('def inner_J')[-1] \
        or 'Pkg 1.2.3 / demo' in num_func
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -k j_kernels -v`
Expected: FAIL (no J docstrings yet).

- [ ] **Step 3: Add `source_map` param + annotate scalar J append**

Change the signature `def print_inner_J(var_address, PARAM, jac, nstep):` to:
```python
def print_inner_J(var_address, PARAM, jac, nstep, source_map=None):
    source_map = source_map or {}
```
Ensure `from Solverz.equation.source import format_source` is importable in this module (added in Task A4).

At the scalar append site, change:
```python
            code_sub_inner_J_blocks.append(pycode(fd1, fully_qualified_modules=False))
```
to:
```python
            _doc = (f"d({eqn_name})/d({var.name})"
                    f"{format_source(source_map.get(eqn_name))}")
            code_sub_inner_J_blocks.append(
                _with_docstring(pycode(fd1, fully_qualified_modules=False), _doc))
```

- [ ] **Step 4: Annotate the loop-jac kernel append**

At the LoopEqn block append site (where `block_source` is appended to `mut_mat_block_funcs`), change:
```python
        mut_mat_block_funcs.append(block_source)
```
to:
```python
        _doc = (f"d({eqn_name})/d({var.name})"
                f"{format_source(source_map.get(eqn_name))}")
        mut_mat_block_funcs.append(_with_docstring(block_source, _doc))
```

- [ ] **Step 5: Pass `source_map` from `render_modules`**

In `Solverz/code_printer/python/module/module_generator.py`, just before the `J = print_inner_J(...)` call (line 98), add:
```python
    source_map = {n: getattr(e, 'source', None) for n, e in eqs.EQNs.items()}
```
and change the call to:
```python
    J = print_inner_J(eqs.var_address,
                      eqs.PARAM,
                      eqs.jac,
                      eqs.nstep,
                      source_map=source_map)
```

- [ ] **Step 6: Run to verify it passes**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -k j_kernels -v`
Expected: PASS.

- [ ] **Step 7: Regression**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest Solverz/code_printer -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Solverz/code_printer/python/module/module_printer.py Solverz/code_printer/python/module/module_generator.py tests/test_provenance.py
git commit -m "feat(printer): docstring each J kernel with d(eqn)/d(var) + source"
```

---

### Task A6: module docstring (version + timestamp + provenance + table)

**Files:**
- Modify: `Solverz/code_printer/python/module/module_generator.py` (`render_modules`: build `manifest`, stash in `code_dict`; `print_init_code`: new `manifest` param + docstring builder; the `render_as_modules` call site that invokes `print_init_code`)
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_provenance.py
import re, datetime


def test_module_docstring_has_version_timestamp_provenance_table():
    mod = _render(_demo_model_with_source(), jit=False)
    init = _read(mod, '__init__.py')
    assert 'Auto-generated by Solverz' in init
    # ISO-8601 UTC timestamp on the header line.
    assert re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00', init)
    assert 'Provenance:' in init
    assert 'Pkg 1.2.3' in init
    assert 'demo' in init
    # Equation table maps a function to its equation + row range.
    assert 'inner_F0' in init and 'e0' in init


def test_module_docstring_groups_unsourced_as_user_defined():
    m = Model()
    m.x = Var('x', np.ones(1))
    m.b = Param('b', np.array([1.0]))
    m.e = Eqn('e', m.x[0] - m.b[0])
    init = _read(_render(m), '__init__.py')
    assert 'Auto-generated by Solverz' in init
    assert '(user-defined)' in init


def test_num_func_is_timestamp_free_and_stable():
    """num_func.py carries no timestamp, so it is byte-stable across renders."""
    m = _demo_model_with_source()
    a = _read(_render(m), 'num_func.py')
    b = _read(_render(_demo_model_with_source()), 'num_func.py')
    assert a == b
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -k module_docstring -v`
Expected: FAIL (docstring still the one-liner).

- [ ] **Step 3: Build the manifest in `render_modules`**

In `module_generator.py`, inside `render_modules`, after `source_map = {...}` (added in Task A5) add:

```python
    # Provenance manifest for the module docstring: one row per F
    # equation with its function name, declaration-order row range, and
    # source. eqs.a maps eqn name -> row slice.
    manifest = []
    for _idx, (_eqn_name, _eqn) in enumerate(eqs.EQNs.items()):
        _addr = eqs.a[_eqn_name]
        _start = getattr(_addr, 'start', None)
        _stop = getattr(_addr, 'stop', None)
        manifest.append({
            'func': f'inner_F{_idx}',
            'eqn': _eqn_name,
            'rows': (_start, _stop),
            'source': getattr(_eqn, 'source', None),
        })
    code_dict['manifest'] = manifest
```

Note: if `eqs.a[_eqn_name]` is not a Python `slice` (no `.start`/`.stop`), adapt to the address object's accessor (e.g. `eqs.a.address[_eqn_name]`); confirm by printing one entry during Step 6.

- [ ] **Step 4: Rewrite `print_init_code` to render the docstring**

Replace the docstring-building head of `print_init_code` (currently lines 277-281) and add a `manifest=None` parameter:

```python
def print_init_code(eqn_type: str, module_name, eqn_param, manifest=None):
    import datetime
    from ...._version import __version__
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')

    lines = [f'Auto-generated by Solverz {__version__} on {ts}.', '']

    manifest = manifest or []
    # Provenance summary: distinct package+version -> set of components.
    prov = {}
    for row in manifest:
        s = row.get('source')
        if s:
            key = f"{s.get('package', '?')} {s.get('version', '?')}"
            prov.setdefault(key, set()).add(s.get('component', '?'))
        else:
            prov.setdefault('(user-defined)', set())
    if prov:
        lines.append('Provenance:')
        for key in sorted(prov):
            comps = ', '.join(sorted(c for c in prov[key] if c))
            lines.append(f'  {key}' + (f'   components: {comps}' if comps else ''))
        lines.append('')

    if manifest:
        lines.append('Equations:')
        lines.append(f"  {'function':<12}{'equation':<24}{'rows':<14}source")
        for row in manifest:
            r0, r1 = row['rows']
            rng = f'[{r0}:{r1}]'
            s = row.get('source')
            src = (f"{s.get('package', '?')} {s.get('version', '?')} / "
                   f"{s.get('component', '?')}") if s else '(user-defined)'
            lines.append(f"  {row['func']:<12}{row['eqn']:<24}{rng:<14}{src}")
        lines.append('')

    code = '"""\n' + '\n'.join(lines) + '\n"""\n'
    code += 'from .num_func import F_, J_\n'
    code += 'from .dependency import setting, p__ as p, y__ as y\n'
    code += 'import time\n'
```

(The rest of `print_init_code`, from `match eqn_type:` onward, is unchanged.)

- [ ] **Step 5: Thread `manifest` into the `print_init_code` call**

`print_init_code` is invoked inside `render_as_modules`. Locate that call (grep `print_init_code(` in `module_generator.py`) and add the manifest argument, sourcing it from `code_dict` which `render_as_modules` already receives:

```python
    initiate_code = print_init_code(eqn_type, module_name, eqn_parameter,
                                    manifest=code_dict.get('manifest'))
```

(If `render_as_modules` does not currently take `code_dict`, it does — it builds the module code from it; reuse the same parameter. Confirm the local variable name for the module/`code_dict` argument and use it.)

- [ ] **Step 6: Run to verify it passes**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -k module_docstring -v`
Expected: PASS (3 tests). If `test_module_docstring_*` fails on row ranges, print `eqs.a[_eqn_name]` once to confirm the accessor and adjust Step 3.

- [ ] **Step 7: Commit**

```bash
git add Solverz/code_printer/python/module/module_generator.py tests/test_provenance.py
git commit -m "feat(printer): module docstring with version, timestamp, provenance table"
```

---

### Task A7: end-to-end + regression + determinism

**Files:**
- Test: `tests/test_provenance.py` (mixed-source model)

- [ ] **Step 1: Write the mixed-model + jit test**

```python
# append to tests/test_provenance.py
def test_mixed_sourced_and_unsourced_model():
    m = Model()
    m.x = Var('x', np.ones(2))
    m.b = Param('b', np.array([1.0, 2.0]))
    m.comp = Eqn('comp', m.x[0] - m.b[0])
    m.comp.source = {'component': 'gt', 'package': 'SolMuseum', 'version': '0.2.0'}
    m.user = Eqn('user', m.x[1] - m.b[1])  # no source
    init = _read(_render(m), '__init__.py')
    assert 'SolMuseum 0.2.0' in init
    assert '(user-defined)' in init


def test_jit_render_still_works_with_docstrings():
    # Docstrings must not break the Numba @njit path.
    mod = _render(_demo_model_with_source(), jit=True)
    init = _read(mod, '__init__.py')
    assert 'Auto-generated by Solverz' in init
```

- [ ] **Step 2: Run to verify it passes**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/test_provenance.py -v`
Expected: PASS (all provenance tests).

- [ ] **Step 3: Full Solverz regression**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/ Solverz/ -q`
Expected: PASS (the single pre-existing `test_rodas_event.py::test_orbit` failure noted in repo history is unrelated; everything else passes). Investigate any NEW failure before continuing.

- [ ] **Step 4: Commit**

```bash
git add tests/test_provenance.py
git commit -m "test(provenance): mixed-source and jit end-to-end coverage"
```

---

### Task A8: release note

**Files:**
- Modify: `docs/src/release_notes.md` (new section at top)

- [ ] **Step 1: Add a release-note entry**

Add a new top section describing: generated modules now carry a docstring with the generating Solverz version + timestamp + an equation/provenance table; each F equation function and J derivative kernel is docstring-annotated with its equation (and source component/package/version when stamped via the new `stamp_source` helper); plain models degrade to name-only docstrings; `num_func.py` stays byte-stable (only the `__init__.py` header timestamp varies). Mention the public API: `from Solverz import stamp_source`.

- [ ] **Step 2: Commit**

```bash
git add docs/src/release_notes.md
git commit -m "docs: release note for generated-module provenance docstrings"
```

---

# PART B — SolMuseum stamping

> Depends on Part A landing in the editable Solverz install (it already is, since `sim` points at `Solverz-dev`).

### Task B1: stamp every SolMuseum component

**Files:**
- Modify: every component module under `SolMuseum/SolMuseum/ae/` and `SolMuseum/SolMuseum/dae/` that defines a `.mdl()`.
- Test: `SolMuseum/SolMuseum/<tests>/test_provenance.py`

- [ ] **Step 1: Enumerate the components**

Run (lists each `.mdl(` definition and the `rename_mdl(` call that marks the end of model assembly):
```bash
cd /Users/ruizhiyu/Dropbox/dev/SolMuseum
grep -rnE "def mdl\(|rename_mdl\(" SolMuseum/ | sort
```
Record the file + the component type name for each (the class name, e.g. `gt`, `pv`, `st`, `eb`, `eps_network`, `heat_network`, `gas_network`).

- [ ] **Step 2: Write the failing test**

```python
# SolMuseum/SolMuseum/<tests>/test_provenance.py
import os, sys, tempfile, uuid
import numpy as np
from Solverz import module_printer


def _render_init(model):
    spf, y0 = model.create_instance()
    d = tempfile.mkdtemp(prefix='sm_prov_')
    name = f'prov_{uuid.uuid4().hex[:8]}'
    module_printer(spf, y0, name, directory=d, jit=False).render()
    with open(os.path.join(d, name, '__init__.py')) as f:
        return f.read()


def test_eps_network_stamps_solmuseum_version():
    # Build a small eps_network model (reuse the case30 mock from the
    # cookbook bench, or the smallest available fixture) and render it.
    from SolMuseum._version import __version__ as sm_ver
    from SolMuseum.ae import eps_network
    pf = _make_test_pf()          # smallest available power-flow fixture
    m = eps_network(pf).mdl(dyn=False, loopeqn=True)
    init = _render_init(m)
    assert f'SolMuseum {sm_ver}' in init
    assert 'eps_network' in init
```
(Use the package's existing test fixtures for `_make_test_pf`; mirror an existing SolMuseum ae test for the smallest network setup.)

- [ ] **Step 3: Run to verify it fails**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest SolMuseum/.../test_provenance.py -v`
Expected: FAIL (no `SolMuseum <ver>` in docstring yet).

- [ ] **Step 4: Add the stamp to each component**

In each component module, add the import once near the top:
```python
from Solverz import stamp_source
from SolMuseum._version import __version__ as _sm_version
```
and inside each `.mdl()`, immediately before `return m` (right after the existing `rename_mdl(m, name)` when present), add:
```python
    stamp_source(m, component='<COMPONENT_TYPE>', package='SolMuseum', version=_sm_version)
```
where `<COMPONENT_TYPE>` is the class/type name from Step 1 (e.g. `'gt'`). For composite components that aggregate sub-component models (e.g. one that `model.add(...)`s others), keep `overwrite=False` (the default) so sub-stamps survive; stamp the composite last.

- [ ] **Step 5: Run to verify it passes**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest SolMuseum/.../test_provenance.py -v`
Expected: PASS.

- [ ] **Step 6: SolMuseum regression**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest SolMuseum/ -q`
Expected: PASS (stamping is metadata-only; no numerical change). Note: some SolMuseum tests need `ipopt`; skip/xfail per the repo's existing pattern.

- [ ] **Step 7: Commit (in the SolMuseum repo, on a feature branch)**

```bash
cd /Users/ruizhiyu/Dropbox/dev/SolMuseum
git checkout -b feat/provenance-stamping
git add SolMuseum/ ; git commit -m "feat: stamp component provenance via Solverz stamp_source"
```

---

# PART C — SolPSDyn stamping

> Same pattern as Part B, applied to SolPSDyn components and the system wiring. SolPSDyn is private; land per its own process.

### Task C1: stamp every SolPSDyn component + wiring

**Files:**
- Modify: every component under `SolPSDyn/SolPSDyn/dae/` (`genrou`, `exst1`, `esst3a`, `avr1`, `esdc2a`, `tgov1`, `ieeeg1`, `gt`, `ieeest`, `st2cut`, `pss1`, `pss2`) and the wiring in `SolPSDyn/SolPSDyn/system.py`.
- Test: SolPSDyn test dir `test_provenance.py`.

- [ ] **Step 1: Enumerate components + wiring sites**

```bash
cd /Users/ruizhiyu/Dropbox/dev/SolPSDyn
grep -rnE "def mdl\(|make_namers\(|def compile\(|Eqn\('wire" SolPSDyn/ | sort
```
Record each component `.mdl()` and the wiring-equation creation in `system.py`.

- [ ] **Step 2: Write the failing test**

```python
# SolPSDyn/.../test_provenance.py
import os, tempfile, uuid
import numpy as np
from Solverz import module_printer
from SolPSDyn._version import __version__ as ps_ver
from SolPSDyn.dae.genrou import genrou


def test_genrou_stamps_solpsdyn_version():
    g = genrou(name='m1', ux=np.array([1.0]), uy=np.array([0.0]),
               Pg=np.array([0.4]), Qg=np.array([0.0]), H=np.array([6.5]))  # mirror test_genrou_equilibrium fixture
    m = g.mdl(rename=True)
    spf, y0 = m.create_instance()
    d = tempfile.mkdtemp(prefix='ps_prov_')
    name = f'prov_{uuid.uuid4().hex[:8]}'
    module_printer(spf, y0, name, directory=d, jit=False).render()
    with open(os.path.join(d, name, '__init__.py')) as f:
        init = f.read()
    assert f'SolPSDyn {ps_ver}' in init
    assert 'genrou' in init
```
(Use the exact `genrou(...)` constructor args from `SolPSDyn/SolPSDyn/dae/test/test_genrou_equilibrium.py`.)

- [ ] **Step 3: Run to verify it fails**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest SolPSDyn/.../test_provenance.py -v`
Expected: FAIL.

- [ ] **Step 4: Add the stamp to each component + wiring**

In each component module add near the top:
```python
from Solverz import stamp_source
from SolPSDyn._version import __version__ as _ps_version
```
and before each `.mdl()` returns its model, after the `make_namers`/rename step, add:
```python
    stamp_source(m, component='<COMPONENT_TYPE>', package='SolPSDyn', version=_ps_version)
```
In `system.py`, after the wiring equations are added to the assembled model and before returning it from `compile()`, stamp the remaining unsourced (wiring) equations:
```python
    stamp_source(mdl, component='wiring', package='SolPSDyn', version=_ps_version)  # overwrite=False keeps component stamps
```

- [ ] **Step 5: Run to verify it passes**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest SolPSDyn/.../test_provenance.py -v`
Expected: PASS.

- [ ] **Step 6: SolPSDyn regression**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest SolPSDyn/ -q`
Expected: PASS.

- [ ] **Step 7: Commit (SolPSDyn repo, feature branch)**

```bash
cd /Users/ruizhiyu/Dropbox/dev/SolPSDyn
git checkout -b feat/provenance-stamping
git add SolPSDyn/ ; git commit -m "feat: stamp component + wiring provenance via Solverz stamp_source"
```

---

# Final integration check

- [ ] **Compose SolMuseum + SolPSDyn end-to-end**

Build a small system that uses a SolMuseum network plus a SolPSDyn machine (mirror `SolPSDyn/SolPSDyn/system.py`'s example), render it, and confirm the module docstring lists BOTH `SolMuseum <ver>` and `SolPSDyn <ver>` in the Provenance block and that per-function docstrings show the right component per equation. Capture the generated `__init__.py` header in the PR description.

- [ ] **Cross-repo smoke**

Run: `/opt/miniconda3/envs/sim/bin/python -m pytest tests/ Solverz/ -q` (Solverz) and the SolMuseum/SolPSDyn provenance tests once more together.
