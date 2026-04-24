# Solverz Development Guidelines

## Project Overview

Solverz is a Python-based simulation modelling language that provides symbolic modelling interfaces and numerical solvers for algebraic equations (AE), ordinary differential equations (ODE), and differential-algebraic equations (DAE).

## Canonical documentation URLs

Use these URLs whenever linking to Solverz docs from code, tests, comments, commit messages, PR descriptions, or release notes. Do **not** link to the underlying ReadTheDocs hostnames.

- **Solverz reference docs**: <https://docs.solverz.org/>
- **Solverz Cookbook (worked examples)**: <https://cookbook.solverz.org/latest/>

## Development Conventions

- Run tests with: `pytest tests/ Solverz/`
- Use conventional commit format: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`
- Prefer composition over inheritance
- Code style follows existing patterns in the codebase

## Architecture Notes

- **Symbolic layer** (`Solverz/sym_algebra/`): SymPy-based symbolic expressions, functions, and matrix calculus
- **Equation layer** (`Solverz/equation/`): equation definitions, Jacobian assembly, parameter management
- **Code printer** (`Solverz/code_printer/`): generates Python/Numba code for F and J functions
  - `inline/`: direct lambdify (no Numba)
  - `module/`: file-based module generation with optional `@njit`
- **Solvers** (`Solverz/solvers/`): Newton-Raphson, ODE/DAE integrators

### Matrix-Vector Equations

Use `Mat_Mul(A, x)` for matrix-vector products in equations. The matrix parameter should be declared with `dim=2, sparse=True`. The matrix calculus module automatically computes symbolic Jacobians. See `docs/src/matrix_calculus.md` for details.

`MatVecMul` is a legacy interface — new code should use `Mat_Mul`.

## Release Process

- Add release notes to `docs/src/release_notes.md` every time a new version is to be released
- Document new features, breaking changes, deprecations, and bug fixes

### Multi-repo coordinated release (Solverz + SolMuseum + Solverz-Cookbook)

Three repositories form a dependency chain:

```
Solverz   (PyPI, upstream)
    ^
    |  Solverz>=X,<Y          [pyproject.toml, install-time dep]
    |
SolMuseum (PyPI)
    ^
    |  Solverz>=X, SolMuseum>=Z   [docs/requirements.txt]
    |
Cookbook  (ReadTheDocs only — no wheel, no PyPI)
```

Each of Solverz and SolMuseum publishes to PyPI on tag push via
`pypa/gh-action-pypi-publish` and gates `publish-to-pypi` on its own
CI (Solverz additionally gates on `tests_in_museum` /
`tests_in_cookbook`, which check out those repos' `main` and install
the tagged commit via git URL).

**Chicken-and-egg to be aware of:**

- SolMuseum's own CI installs Solverz from PyPI. A SolMuseum `main`
  that uses new Solverz API cannot pass its own CI until the new
  Solverz is on PyPI.
- Solverz's `tests_in_museum` job checks out SolMuseum `main` (not
  the PR branch) and installs the tagged Solverz commit by git URL.
  Before the downstream PRs merge, the old SolMuseum `main` runs
  against the new Solverz — this passes if there is no breaking
  Solverz API change, fails if there is.

**Release the upstream first, then cascade.** Do not try to merge
or tag all three at once — cross-repo CI will deadlock.

#### Phase 0 — Preparation

1. Pick version numbers. Default scheme:
   - Solverz next patch: `0.X.Y`
   - SolMuseum next patch: `0.a.b`
   - Cookbook tag: `v0.X.Y` (mirrors Solverz major.minor)
2. In Solverz-dev, add a new section at the top of
   `docs/src/release_notes.md` describing new features / breaking
   changes / deprecations / bug fixes.
3. Verify **own-repo** CI is green on each feature branch. Cross-repo
   failures on the Solverz PR are expected at this stage and do not
   block merge.
4. Local cross-repo smoke test: in one virtualenv,
   `pip install -e` all three editable, run representative tests
   (`pytest tests/ Solverz/` in Solverz, the Cookbook `test_ies.py`
   and `test_pf_jac.py`, and any touched SolMuseum module tests).

#### Phase 1 — Solverz release

1. `cd Solverz-dev && git checkout main && git pull --ff-only`.
2. Merge the Solverz PR to `main` (squash or merge-commit). Admin
   merge is acceptable if cross-repo CI is red solely because
   downstream PRs haven't landed yet. Confirm the merge is
   backward-compatible for SolMuseum's current `main` before
   overriding.
3. `git pull --ff-only` to fetch the merge commit.
4. Annotate-tag: `git tag -a 0.X.Y -m "Release 0.X.Y"`.
5. `git push origin 0.X.Y`. This triggers the tag-push CI:
   `built_in_tests` → `tests_in_museum` / `tests_in_cookbook` →
   `build` → `publish-to-pypi` → `github-release`.
6. If `tests_in_museum` / `tests_in_cookbook` fail due to flaky /
   tolerance / environmental issues (not real API break), use
   `gh run rerun <id> --failed` to retry. If they fail for a real
   reason, delete the tag (`git push --delete origin 0.X.Y`) and
   investigate — do not bypass the gate.
7. Verify PyPI: `pip index versions Solverz` reports `0.X.Y`. Wait
   1–3 min for CDN propagation before starting Phase 2.

#### Phase 2 — SolMuseum release

1. `cd SolMuseum`. On the feature branch, bump the Solverz floor in
   `pyproject.toml` to `Solverz>=0.X.Y,<0.{X+1}` if the new release
   added API the SolMuseum branch relies on. Commit + push.
2. Merge the SolMuseum PR to `main`. Its push-CI (`run-tests`) now
   installs Solverz `0.X.Y` from PyPI and should pass.
3. `git checkout main && git pull --ff-only`.
4. `git tag -a 0.a.b -m "Release 0.a.b"` and push the tag.
5. Tag-push CI publishes to PyPI. Verify with
   `pip index versions SolMuseum`.

#### Phase 3 — Cookbook release

Cookbook has no wheel; the tag is solely the ReadTheDocs version
cut. Its feature PRs are opened against `dev`, not `main`.

1. `cd Solverz-Cookbook`. On the Cookbook feature branch, bump
   `docs/requirements.txt`:
   - `Solverz>=0.X.Y`
   - `SolMuseum>=0.a.b`
   Commit + push.
2. Merge the feature branch into `dev`.
3. Open a `dev → main` PR; once CI is green, merge it.
4. `git checkout main && git pull --ff-only`.
5. `git tag -a v0.X.Y -m "Release v0.X.Y"` and push the tag.
6. ReadTheDocs auto-builds the new version and adds it to the
   version switcher at <https://cookbook.solverz.org/>.

#### Phase 4 — Post-release verification

1. Clean venv smoke test:
   ```
   pip install "Solverz==0.X.Y" "SolMuseum==0.a.b"
   python -c "from Solverz import Set; from SolMuseum.dae import heat_network; print('ok')"
   ```
2. Confirm the new Cookbook version appears in the ReadTheDocs
   version switcher.
3. Skim the auto-generated GitHub release pages for Solverz and
   SolMuseum; paste short highlights in the release body if
   desired.

#### Rollback

- PyPI is append-only. Use `pip yank Solverz==0.X.Y --reason "..."`
  to hide a bad release; ship a `0.X.Y+1` patch with the fix.
  Do not delete a published git tag — downstream consumers may
  have cached the reference.

#### Known limitation

The current Solverz CI gates `publish-to-pypi` on
`tests_in_museum` / `tests_in_cookbook`, which run against the
**main** of those downstream repos. When a coordinated release is
in flight, these jobs may require reruns or, in pathological
cross-API cases, a short-lived workflow commit that temporarily
drops them from `needs:` on the publish job. Prefer a rerun over
editing CI.
