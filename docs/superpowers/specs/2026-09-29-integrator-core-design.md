# Design: the Solverz integrator core, milestone C2

Date: 2026-09-29; revision 2 of 2026-09-30, after five reviews
Status: specification for implementation
Branch: `feat/integrator-core`, based on `056e87a` (PR #188, the residual contract `F(t, y, p, out=None)`)
Tracking issue: smallbunnies/Solverz#189
Governing documents: the design document of the project lead, whose location the operations notes record (its section "2026-09-29修订" overrides the rest), the binding decisions D1 to D8 of the project lead, restated in Section 2.

## 0. Conventions

Every `path:line` without a prefix refers to the worktree at `056e87a`. `SolPSDyn@c16a041:` refers to SolPSDyn at commit `c16a041828a5022b94569f2ee49be4afa799a6fd`. `ODE@7393799:` refers to OrdinaryDiffEq.jl at master `739379950dc33ccd6bb427931225013aca5486b7`, and `plan:` to the design document. "Byte-equal" means `a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()`; it is stricter than `np.array_equal`, since it also distinguishes `-0.0` from `+0.0`. "Legacy" means the existing solvers under `Solverz/solvers/daesolver/`, and "legacy Rodas" means `Solverz/solvers/daesolver/rodas/rodas.py` at `056e87a`. The two commits `87693f2` and `1b51f48` that change legacy Rodas on the frozen branch `feat/kernel-eqn` are not ancestors of `056e87a` and are not part of any reference in this document.

Publication rule. This document is committed with milestone I8 as the design record of C2. Every reference into two private projects of the author, a data-centre study and a machine-learning study, every absolute path of the laptop or of the server, and the operational detail of Sections 16.1, 17.1 and 17.2 were moved before that commit to the operations notes, `c2-tools/C2_OPERATIONS.md`, outside the repository. This copy keeps every decision and replaces each moved reference by "a private benchmark record" or "the operations notes".

## 1. Purpose and scope

### 1.1 Purpose

Each legacy DAE solver carries its own copy of step control, acceptance and rejection, event location, dense output, output buffers, `Stats`, the progress bar, the KLU cache and row equilibration, and the copies disagree (plan:23). C2 creates one stepping loop that owns all of this, in the division of work of OrdinaryDiffEq.jl, and one `perform_step` per algorithm. The Rosenbrock family comes first, with the coefficient tables that legacy Rodas already uses. An algorithm author writes only how one step is taken, in one of three styles, and the core supplies everything else.

### 1.2 In scope

1. The package `Solverz/integrator/` (Section 3) with the Integrator, the controllers, the linear-algebra service, the two `dF/dt` policies, saving, callbacks and events, the author contract, the conformance kit, and the algorithms `Rodas3`, `Rodas4`, `Rodasp`, `Rodas5P`, `ImplicitEuler` and `Trapezoid`.
2. A legacy-compatible configuration of the Rosenbrock algorithms that reproduces legacy Rodas trajectories bit for bit on event-free runs (D3b, Section 6).
3. A `DeprecationWarning` in legacy Rodas and nothing else in legacy code (D1, Section 14).
4. One addition outside `Solverz/integrator/` that changes no existing behaviour: the method `klu_decomposition.solve_into(b, out)` in `Solverz/solvers/klu_backend.py` (Section 8.2).
5. The tests under `tests/integrator/` (Section 15), the user guide `docs/src/integrator.md`, the author guide `docs/src/integrator_adding_algorithms.md`, reference entries and a release note (Section 18, milestone I8).
6. A read-only gate run of SolPSDyn tests at `c16a041` (Section 16) and the benchmarks of D7 (Section 17), both on the server.

### 1.3 Out of scope

- Any change of behaviour of a legacy solver. Legacy Rodas gains only the warning; no other legacy solver gets a warning in C2 (D1).
- The unified build entry point, which is cancelled. `made_numerical`, `module_printer`, the printers and `nDAE` are unchanged (D2).
- An in-place `J_` and an in-place assembly of the iteration matrix. `CooToCsc` builds a new `csc_array` at every call (`Solverz/num_api/custom_function.py:264-274`), and the iteration matrix keeps the SciPy operation chain of legacy Rodas (Section 8.2), whose output KLU receives unchanged.
- Radau IIA, a shared Newton solver with Jacobian ageing, W-methods, a predictive controller, table-driven explicit Runge-Kutta or SDIRK families, a first-same-as-last hand-off of the end residual, composite algorithms, matrix-free iteration matrices and automatic differentiation. Explicit and SDIRK methods can be written in the formula style (Section 12.3). The simplified Newton of Section 12.4 serves only `implicit()` in C2.
- Integration backwards in time. `t0 > tend` raises `ValueError`, as legacy does (`rodas.py:74-75`).
- Reproducing legacy event semantics. They are defective (Appendix B) and D3 excludes them.
- Changes to SolPSDyn, SolAlg, SolMuseum, the Cookbook or `EventLoop`. The option C6 of the design document, moving the segment logic of `EventLoop` into the core, is not part of C2.
- A `maxiters` limit and any capacity limit on the output.

## 2. Binding decisions and flagged items

### 2.1 Decisions of the project lead

D1 to D8 are binding. Where this document refines one, the refinement is marked and repeated in Section 19. In summary: legacy solvers are never deleted or changed, and legacy Rodas gains a `DeprecationWarning` that points at the caller and names only its replacements (D1). The postponed milestones and the solvers they concern are not referred to, and the unified build API is cancelled (D2). Parity is proved per step against a validated transcription (D3a) and per trajectory in a documented legacy-compatible configuration (D3b). The default configuration follows the kernel rules (D4). Events are continuous callbacks located on the interpolant, and discrete callbacks at `tstops` may change the model (D5). Authors extend the core with tables, formulas or in-place code (D6). In-place and out-of-place versions are measured against each other, and the out-of-place Rodas4 exists for the benchmark only (D7). Tests and benchmarks run on the server, and every test passes on SuperLU (D8).

### 2.2 Concrete defects found in binding decisions, with the resolution this document adopts

F1, D4 "KLU analysis reused across calls through `model_cache(dae)`". Reusing across calls whatever `model_cache(dae)` holds makes the result of a call depend on the calls made before it, in two cases. First, the `superlu` slot of the same cache holds the SuperLU column ordering (`Solverz/solvers/klu_backend.py:211-226`). With a cached ordering SuperLU factorizes `A Pc` with `permc_spec='NATURAL'`, and an exact pivot tie can then be broken on another row than under COLAMD (`Solverz/solvers/laesolver.py:178-184`, `:255`), so the first factorization of a second call can differ in the last bits from that of the first call. SuperLU is the only backend on the Windows CI runners (`docs/src/release_notes.md:18`). Second, from `MATCHING_MIN_N = 1000` unknowns up, a KLU analysis contains a maximum-product row matching computed from the values of the matrix that triggered it (`klu_backend.py:63-64, :306-331`), so a later call factorizes with a permutation chosen from another call's values. A CCT bisection on a large model would then depend on the order of its calls, and two identical calls could differ. Resolution: the default configuration shares across calls only a KLU analysis that contains no matching and that the current matching setting would reproduce, and keeps everything else, including the SuperLU ordering, per call (Section 8.4). A call's result then never depends on earlier calls, on either backend and at any size. D4's reuse holds for every model below 1000 unknowns, and above that size whenever the matching is switched off. This needs the lead's confirmation (Section 19, O1).

F2, D5 "a crossing near the start of a step is never discarded" against design rule 9, "`repeat_nudge` of 1 percent of the step" (plan:94). After an event located with `rootfind='left'`, the state lies just before the crossing, and the next step starts there. Without the nudge the same crossing is found again in the first floats of that step and reported twice. Resolution: a re-crossing within the nudge interval, by the component that fired at the start of the step, is the event already reported, not a new one; every other crossing near the start of a step is kept (Section 11.3, step 5). This needs the lead's confirmation (Section 19, O1).

### 2.3 Deviations from the text of the design document

These are corrections of the design document, not of D1 to D8.

- Design rule 4 scales the error by `max(|u|, |uprev|)` and calls it "consistent with legacy" (plan:89, :99). Legacy scales by `|ynew|` only (`rodas.py:208`). The default configuration uses `max(|u|, |uprev|)` (D4), and the legacy-compatible configuration uses `|ynew|` (D3b).
- Design rule 9 reads `interp_points = 10` as ten subintervals (plan:94). The name and the value come from SciML, where it means ten sample points and nine subintervals (`ODE@7393799:lib/DiffEqBase/src/callbacks.jl:537-561`). This document follows SciML.
- The design document expects the rendered and the inline model to take the same steps with `max|dy| = 0` (plan:171). The recorded data contradict that as a general rule. On the server the 2 s runs of the SDCIB model of a private benchmark gave `max|dy| = 0` at every tolerance, while on the laptop the same run gave `1.12e-08` at `rtol = 1e-8`, with the unchanged printer too (a private benchmark record). Over 12 s the step counts separate, 4835 against 4814, and the trajectories differ by `2.79e-04` near a finite-time pole (a private benchmark record). Numba and NumPy evaluate `**` and transcendental functions with different routines. Section 15 states the equality that is actually required, one criterion used everywhere. The kit's rendered check is opt-in (`rendered=True`), where plan:131 lists it among the required checks, since it needs a Numba compilation per algorithm.
- The design document puts the KLU cache on `model_cache(dae)` for Rodas (plan:51, :66, :99) while legacy Rodas creates a fresh `KLUCache` per call (`rodas.py:124`). The default configuration shares the KLU analysis across calls under the rule of F1, and the legacy-compatible configuration uses a fresh cache per call (D3b).
- The design document's IController text says "拒绝后一次不增长" (plan:63). Legacy keeps the growth cap at 1 until the next accepted step, not for one attempt (`rodas.py:349, :354`). Both controllers of Section 7 reproduce the legacy meaning.
- `repeat_step` is dropped from the `perform_step` signature (plan:108). No C2 algorithm repeats a step, and SciML uses the argument only for FSAL bookkeeping.
- The design's author contract lists the traits `adaptive_order`, `is_fsal`, `is_implicit` and `extrapolates` (plan:115). This document keeps the traits an algorithm must declare for the core to act correctly: `error_order` replaces `adaptive_order` with a self-describing meaning (Section 12.1), `explicit` replaces `is_implicit`, and `interp_order` states the accuracy of the interpolant for the kit. `is_fsal` and `extrapolates` are dropped with the FSAL hand-off (Section 1.3).
- Plan revision item 2 gives a legacy solver a `DeprecationWarning` in the milestone in which it gains a kernel counterpart. C2 ships `ImplicitEuler` and `Trapezoid`, which resemble the legacy `backward_euler` and `implicit_trapezoid`. They are templates of the author guide, not declared replacements, and D1 allows no warning other than on Rodas, so D1 prevails and those two solvers get no warning.
- Plan revision item 4 lists the Rosenbrock, explicit Runge-Kutta and SDIRK families as table-driven. In C2 the Rosenbrock family is the only table-driven family; explicit and SDIRK methods are written in the formula style with the services of Section 12.2. Issue #189 says that a method "of an existing family" is only its table, which in C2 is true of the Rosenbrock family alone.
- The design's `test_events.py` tests `event_duration` (plan:168). D5 excludes it, so `opt.event_duration` is ignored and not tested.

## 3. Package layout

### 3.1 Files

| File | Responsibility | Public names |
|---|---|---|
| `Solverz/integrator/__init__.py` | Re-exports with an explicit `__all__`, so that `from Solverz.integrator import *` leaks no submodule name | see 3.2 |
| `integrator.py` | `Integrator`, `init`, `solve`; the loop of Section 5 | `Integrator`, `init`, `solve` |
| `options.py` | `IntegratorOptions`, a frozen dataclass read once from `Opt` | `IntegratorOptions` |
| `policies.py` | `DefaultPolicy` and `LegacyRodasPolicy`, the two configurations of Section 6, selected once per call | none exported |
| `controllers.py` | `Controller` base, `IController`, `PIController`, `LegacyRodasController` | these four |
| `linalg.py` | `IterationMatrix` and the factorization objects it returns | none exported |
| `derivative.py` | The `dF/dt` policies `'ode23s'` and `'legacy'` | none exported |
| `nlsolve.py` | The simplified Newton behind `implicit()` | none exported |
| `saving.py` | `SolutionBuffer`, `EventLog`, `to_daesol` | none exported |
| `callbacks.py` | `ContinuousCallback`, `DiscreteCallback`, `preset_time_callback`, the private adapter `_LegacyEventCallback`, `find_root` | the first three |
| `algorithm.py` | `Algorithm`, `StepContext`, `StepFailure` | `Algorithm`, `StepFailure` |
| `rosenbrock.py` | `RosenbrockTableau`, `Rosenbrock`, `Rodas3`, `Rodas4`, `Rodasp`, `Rodas5P` | these six |
| `implicit_euler.py` | `ImplicitEuler`, the template of the author guide | `ImplicitEuler` |
| `trapezoid.py` | `Trapezoid` | `Trapezoid` |
| `testing.py` | `check_algorithm` | `check_algorithm`, imported as `from Solverz.integrator.testing import check_algorithm` |

### 3.2 `Solverz/integrator/__init__.py`

```python
__all__ = ['solve', 'init', 'Integrator', 'IntegratorOptions',
           'Algorithm', 'StepFailure',
           'Rosenbrock', 'RosenbrockTableau', 'Rodas3', 'Rodas4', 'Rodasp', 'Rodas5P',
           'ImplicitEuler', 'Trapezoid',
           'ContinuousCallback', 'DiscreteCallback', 'preset_time_callback',
           'Controller', 'IController', 'PIController', 'LegacyRodasController']
```

`check_algorithm` is not in `__all__`, so importing the package does not import the model-building code that `testing.py` uses.

Staging, recorded at I1. Until I8, `__all__` lists only the names whose modules exist, in the order above, since a name in `__all__` without a binding breaks the star import; each milestone adds its own names, and `test_api.py` asserts the full list at I8.

### 3.3 Top-level exports

`Solverz/__init__.py` gains one line after line 9: `from Solverz.integrator import Rodas3, Rodas4, Rodasp, Rodas5P, ContinuousCallback, DiscreteCallback`. The names `solve`, `init` and `Integrator` are not exported at the top level. A script that runs `from sympy import *` and then `from Solverz import *` would otherwise lose SymPy's `solve`, and `init` is too generic for a star import. Today `from Solverz import *` binds no name `solve`; it binds 91 names, among them the stray submodule names `rodas` and `utilities` (checked with the worktree on `PYTHONPATH`). The new line adds the six names above and, because `Solverz/__init__.py` has no `__all__`, the package name `integrator`. The name `utilities` is already rebound by `from Solverz.solvers import *` (`Solverz/__init__.py:8`) to `Solverz.solvers.nlaesolver.utilities`; C2 does not change that and nothing in the core reads `Solverz.utilities`.

### 3.4 Import rules

- Modules in `Solverz/integrator/` import at module level only from `numpy`, `scipy`, `math`, `inspect`, `weakref`, `Solverz.solvers.option`, `.stats`, `.solution`, `.parser`, `.laesolver`, `.klu_backend`, `.daesolver.daeic`, `.daesolver.rodas.param`, `.daesolver.rodas.rodas` (for `ntrp1` and `ntrp2` only), `Solverz.num_api.num_eqn` and `Solverz.variable.variables`. `MATCHING_MIN_N` is read as the attribute `klu_backend.MATCHING_MIN_N` at the time of use, never imported by name, since `set_klu_matching` rebinds it. Recorded at I1: the list is exhaustive for the Solverz modules, which is what excludes a cycle and what `test_import.py` checks; standard-library modules are allowed as well, since this document itself needs `dataclasses` (Section 3.1), `warnings` (Section 4.3) and `types` (Section 5.2).
- `testing.py` imports `Model`, `Var`, `Param`, `Ode`, `Eqn`, `made_numerical` and `Opt` inside `check_algorithm`, not at module level. Recorded at I7b: the imports, with `module_printer` for the rendered check, are in the private `_check(alg, names)` that `check_algorithm` calls, which also runs chosen checks alone for the tests.
- No legacy module imports `Solverz.integrator`. Importing any `Solverz.solvers` submodule first runs `Solverz/solvers/__init__.py`, which imports every legacy solver (`Solverz/solvers/__init__.py:1-6`), so the reverse import would be circular.
- The legacy import paths `Solverz.solvers.daesolver.rodas.rodas.{Rodas, dfdt, ntrp1, ntrp2}` and `Solverz.solvers.daesolver.rodas.param.Rodas_param` stay, since `sicnm` imports `Rodas_param` (`Solverz/solvers/nlaesolver/sicnm.py:4`) and a test imports `dfdt` (`tests/test_rendered_F_aliasing.py:71`).

## 4. Public API

### 4.1 Signatures

```python
def solve(dae, tspan, y0, alg=None, opt=None, *, callbacks=(), tstops=()) -> daesol
def init(dae, tspan, y0, alg=None, opt=None, *, callbacks=(), tstops=()) -> Integrator

class Integrator:
    def step(self) -> bool       # advance by one accepted step; False once the run is over
    def solve(self) -> daesol    # run to the end and return the result
    def interp(self, tq, out=None) -> np.ndarray   # dense output inside the last accepted step
    def terminate(self) -> None  # end the run after the current callback, or before the next step
    def model_modified(self) -> None   # after the caller changed u, dae.p or dae.M.data between steps

class Algorithm:
    def __call__(self, dae, tspan, y0, opt=None) -> daesol   # legacy-shaped call
```

- `alg=None` means `Rodas4()`. `opt=None` means `Opt()`.
- `tspan` has the legacy meaning. `len(tspan) == 2` saves every accepted step. `len(tspan) > 2` saves at `tspan[1:]` by interpolation and does not change the step sequence. `tspan[0]` is always the first row.
- `callbacks` is a sequence of `ContinuousCallback` and `DiscreteCallback`. `opt.event` adds the adapter of Section 11.6 after them.
- `tstops` is a sequence of times at which a step must end exactly. Values outside the open interval `(t0, tend)` are ignored, as in SciML (`ODE@7393799:lib/OrdinaryDiffEqCore/src/solve.jl:1179-1198`).
- `Rodas4()(dae, tspan, y0, opt)` is `solve(dae, tspan, y0, alg=Rodas4(), opt=opt)`. It has the form of legacy `Rodas(dae, tspan, y0, opt)`, which is what `EventLoop` calls (`SolPSDyn@c16a041:SolPSDyn/dae/event_loop.py:448`), so `EventLoop(dae, solver=Rodas4())` works unchanged.
- `init(...).solve()`, `solve(...)` and `alg(...)` run the same code and give byte-equal results. `solve` and `Algorithm.__call__` each build the `Integrator` directly and never call one another, so each call issues the warning of Section 4.3 at most once. `step()` and `solve()` share one loop body, so the two paths also agree on the tiny-step `tstops` rule, unlike SciML (`ODE@7393799:lib/OrdinaryDiffEqCore/src/iterator_interface.jl:2-27`).
- `solve` and `Algorithm.__call__` print `f"Time elapsed: {end - start}s"` when `opt` is not `None` and `opt.profile` is true, the text of `Solverz/solvers/parser.py:96`.
- `model_modified()` is the only supported way to change the state or the model between two `step()` calls. The caller changes `integ.u`, an array of `dae.p` or `dae.M.data`, or rebinds `dae.p` or `dae.M`, and then calls it. It runs the modification protocol of Section 11.7 at the current `t`. Afterwards `interp` accepts only `tq == t` until the next accepted step, because the change may have overwritten the end state of the step (Section 5.4).

### 4.2 From `Opt` to `IntegratorOptions`

`IntegratorOptions.from_opt(opt, alg, tspan)` reads `Opt` once and never writes to it. The fields not listed here are not read.

| `Opt` field (`Solverz/solvers/option.py:5-29`) | `IntegratorOptions` field | Meaning |
|---|---|---|
| `rtol`, `atol` | `rtol`, `atol` | tolerances; `atol` may be an array of length `n` |
| `f_savety` | `safety` | controller safety factor |
| `fac1` | `qmin` | smallest step factor, `dtnew >= qmin*dt` |
| `fac2` | `qmax` | largest step factor after an accepted step |
| `facmax` | `qmax_init` | largest step factor for the first attempt of the call, the value legacy reads before its first write (`rodas.py:215`) |
| `hinit` | `dt0` | initial step; with `fix_h`, the fixed step |
| `hmax` | `dtmax` | `hmax` if not `None`, else `np.abs(tend - t0)`, which legacy writes into `opt` (`rodas.py:76-77`) and the core does not |
| `fix_h` | `adaptive` | `adaptive = alg.adaptive and not opt.fix_h` |
| `linsolver` | `linsolver` | `resolve_backend(getattr(opt, 'linsolver', None))`, resolved once per call as legacy does (`rodas.py:121`, `Solverz/solvers/laesolver.py:79-85`) |
| `event` | adapter callback | Section 11.6 |
| `pbar` | `pbar` | `tqdm` progress bar, advanced by `t - tprev` per accepted step |
| `profile` | read by `solve` and `__call__` only | timing print of Section 4.1 |
| `scheme` | not read | only compared with the algorithm, Section 4.3 |
| `event_duration` | not read | a crossing near the start of a step is never discarded (D5) |

Further fields: `legacy_compat = alg.legacy_compat`; `dense = len(tspan) > 2`; `failfactor = 2.0`; `max_consecutive_reject = 100`, the legacy limit (`rodas.py:140`); `t0`, `tend` and the saveat nodes.

Recorded at I1. `from_opt` applies the `tspan` rule of the configuration (Section 5.2, step 3), because `t0` is a field and `LegacyRodasController` needs it in the `tspan` dtype (Section 7.3): in the legacy-compatible configuration `t0`, `tend`, the nodes `saveat` and `dtmax` keep the types legacy computes them with, and `dt0` is `hinit` as given; in the default configuration `t0`, `tend`, `dt0` and `dtmax` are Python floats, which carry the bits of `np.float64` at a lower cost per scalar operation, and `saveat` is float64. `policy.prepare_tspan` then reads these fields. The nodes are a view of a read-only copy of `tspan`, and an array `atol` or `rtol` is held as a read-only float64 copy, so a later write to the caller's array does not reach the run. `from_opt` also accepts `opt=None` as `Opt()`.

`from_opt` raises `ValueError` for `t0 > tend` with the legacy text `f't0: {t0} > tend: {tend}'`, for `hinit` not `None` and `hinit <= 0`, for a non-increasing `tspan` of more than two entries in the default configuration, and for a non-adaptive run without `hinit`. For an algorithm with `adaptive = False` the text of the last one reads `f"{alg.scheme} has no error estimate (adaptive = False) and runs with the fixed step opt.hinit, which is not set; an algorithm whose perform_step returns an error estimate declares adaptive = True"`; for `opt.fix_h` it reads `"opt.fix_h needs opt.hinit"`. A missing `hinit` with `fix_h` raises `TypeError` inside legacy Rodas (`rodas.py:152, :162`); the core raises `ValueError` before any step, as plan:82 asks. `Integrator.__init__`, which sees the callbacks, raises `ValueError` for `tstops`, including the `tstops` of a `DiscreteCallback`, in the legacy-compatible configuration. These are argument errors, raised before the run starts, and are not failures in the sense of Section 5.8. Legacy clamps a `hinit <= 0` to `hmin` (`rodas.py:111-117`), so the parity of D3b is claimed for `hinit` either `None` or positive.

### 4.3 When `opt.scheme` disagrees with the algorithm

The algorithm object selects the method. `opt.scheme` is not read. `Opt()` sets `scheme='rodas4'` whether or not the caller chose it (`Solverz/solvers/option.py:12`), so a disagreement can be detected only when `opt.scheme != 'rodas4'`. In that case, and when `opt.scheme != alg.scheme`, the helper `_warn_scheme(opt, alg)` issues:

```python
warnings.warn(f"opt.scheme={opt.scheme!r} is ignored; {type(alg).__name__}() integrates with "
              f"{alg.scheme!r}. Pass the algorithm of the method, for example "
              f"Rosenbrock.from_scheme(opt.scheme).", UserWarning, stacklevel=3)
```

`solve`, `init` and `Algorithm.__call__` each call the helper directly, once, before they build the `Integrator`, and none of them calls another public entry, so `stacklevel=3` points at the caller of the entry and the warning appears once per call. Callers that set `scheme='rodas3'`, such as `SolPSDyn@c16a041:SolPSDyn/dae/test/test_ieeex1.py:144, :342` and a private study script that the operations notes cite, therefore learn that `Rodas4()` would silently change their method. `Rosenbrock.from_scheme(name, *, legacy_compat=False)` maps `'rodas3'`, `'rodas4'`, `'rodasp'`, `'rodas5p'` to `Rodas3`, `Rodas4`, `Rodasp`, `Rodas5P`, case-sensitively as legacy (`Solverz/solvers/daesolver/rodas/param.py:8`). `'rodas3d'` raises `ValueError("'rodas3d' defines no dense-output coefficients (param.py:171-200) and is not provided by the integrator core")`, and any other name raises `ValueError(f"unknown Rosenbrock scheme {name!r}")`.

Recorded at I4. `_warn_scheme` is in `algorithm.py`, since `Algorithm.__call__` and the two functions of `integrator.py` call it, and it ships with those entries at I4; its test stays with I5, as Section 15 lists it.

### 4.4 `Vars` in and out

The behaviour equals `dae_io_parser` (`Solverz/solvers/parser.py:82-108`). When `y0` is a `Vars`, the core integrates `y0.array` and converts on return with the same helper, `parse_dae_v` (`parser.py:117-121`): `sol.Y = parse_dae_v(sol.Y, y0.a)`, and `sol.ye = parse_dae_v(sol.ye, y0.a)` when `sol.te is not None`. `Integrator.__init__` records `y0.a` and `to_daesol` applies the conversion, so `init(...).solve()` and `solve(...)` return the same types. With an ndarray `y0` the result holds ndarrays. `parse_dae_v` indexes row 0, so `te`, `ye` and `ie` are `None`, never empty arrays, when no event was recorded.

### 4.5 The result

The result is the legacy `daesol(T, Y, te, ye, ie, stats)` (`Solverz/solvers/solution.py:23-38`).

- `T` is a new 1-D float64 array and `Y` a new C-contiguous `(len(T), n)` float64 array, both built from the saved rows at the end. Neither is a view of a larger buffer.
- `T[0] == t0` and `Y[0]` is the consistent initial state after `DaeIc`.
- In the default configuration `T[-1] == tend` exactly after a successful run, and `T[-1] == te` exactly after a terminal event.
- `te` is float64 of shape `(k,)`, `ye` float64 `(k, n)`, `ie` int64 `(k,)`; all three are `None` when nothing was recorded. Legacy stores `ie` as float64 (`rodas.py:108`); `SolPSDyn@c16a041:SolPSDyn/dae/event_loop.py:238-288` casts it with `.astype(int)` and so accepts either.
- `stats` is a legacy `Stats` (`Solverz/solvers/stats.py:3-13`) with `scheme = alg.scheme`, the counters of Section 5.9, `ret` in `{'success', 'terminated', 'failed'}` and `succeed = ret != 'failed'`. It also carries the instance attribute `ncondition`, and after a failure the instance attribute `t_fail`, the time printed in the failure message, which with an output grid can lie after `T[-1]`. `Stats` has no `__slots__`, so this needs no change to the class, and callers may still assign to `stats.ret`, as a private study script does.

## 5. The Integrator

### 5.1 Fields

| Field | Meaning |
|---|---|
| `dae`, `alg`, `opts`, `policy`, `controller`, `cache`, `linalg`, `stats` | the model, the algorithm, the options, the configuration of Section 6, the controller, the algorithm's cache from `alloc`, the `IterationMatrix`, the `Stats` |
| `ctx` | the `StepContext` of Section 12.2, built once |
| `n`, `M`, `p`, `model_epoch` | size; mass matrix and parameters, read from `dae` at initialization and in every modification protocol; a counter increased by every modification protocol, which caches derived from `M` compare |
| `t0`, `tend`, `t`, `tprev` | times; `t` is the start of the attempt before `loopfooter` and the current time after it |
| `dt`, `dtpropose`, `dt_untruncated` | step of the attempt; step proposed for the next step; step before truncation at a `tstop` |
| `u`, `uprev` | two float64 arrays of length `n` owned by the Integrator and never rebound |
| `t_step`, `u_step`, `dt_step` | the end time, the end state and the length of the last accepted step, which parameterize its interpolant; `u_step` is the same object as `u` until something changes `u` after acceptance (Section 5.4) |
| `EEst`, `q`, `accept_step`, `force_stepfail`, `new_step` | error estimate of the attempt; controller quantity; outcome; failed attempt; first attempt since the last commit or model change |
| `iter`, `nconsecutive_reject` | number of attempts; consecutive rejected attempts |
| `tstops`, `next_step_tstop`, `tstop_target` | min-heap of stop times including `tend`; the current attempt is truncated to land on `tstop_target` |
| `saveat`, `saveat_idx` | the nodes `tspan[1:]` when `opts.dense`, and the index of the next node |
| `continuous_callbacks`, `discrete_callbacks` | the callbacks, the adapter last. Recorded at I6a: `continuous_callbacks` holds one per-run state per continuous callback, which wraps the callback as `.cb` and keeps its bottom values and the components that fired, so that one callback object can serve several calls |
| `sol`, `events` | `SolutionBuffer` of saved rows; `EventLog` of recorded events |
| `terminated`, `failed`, `finished`, `retcode` | run state |
| `_F0`, `_J0`, `_ft`, `_W_cache`, `_nl_eta`, `_interp_ready`, `_interp_valid`, `_daeic_backend`, `_pbar` | per-step service caches of Sections 8, 9 and 12; whether `addsteps` ran for the current step; whether the step's end state is intact (Section 5.4); the global backend at initialization, for `DaeIc` (Section 5.2); the progress bar |

### 5.2 Initialization

`Integrator.__init__(dae, tspan, y0, alg, opt, callbacks, tstops)`:

1. Check the algorithm's style (Section 12.1) and raise `TypeError` if `perform_step` is missing or its signature does not match `alg.inplace`. The result is cached per class.
2. `opts = IntegratorOptions.from_opt(opt, alg, tspan)`, then `policy = LegacyRodasPolicy(...) if opts.legacy_compat else DefaultPolicy(...)`; in the legacy-compatible configuration, `ValueError` if `tstops` or any `DiscreteCallback.tstops` is non-empty.
3. `t0, tend` from `policy.prepare_tspan(tspan)`: `np.array(tspan)` with its own dtype in the legacy-compatible configuration, as `rodas.py:71`, so an integer `tspan` gives integer `t0` exactly as legacy; `np.asarray(tspan, dtype=np.float64)` in the default configuration.
4. If `y0` is a `Vars`, record `y0.a` and take `y0.array`. Copy: `u0 = np.array(y, dtype=np.float64)`. The caller's array is never written. `DaeIc` returns the caller's own array when the initial state is consistent (`Solverz/solvers/daesolver/daeic.py:35-36`), and `dae_io_parser` passes `Vars.array` uncopied (`parser.py:84-88`).
5. `M = dae.M`, `p = dae.p`; the residual service keeps `dae.F` if `'out'` appears in `inspect.signature(getattr(F, 'py_func', F), follow_wrapped=False).parameters`, and otherwise wraps it with `_with_out` (`Solverz/num_api/num_eqn.py:30-44`). With `follow_wrapped=False` an `F` that `nDAE` has already wrapped is recognized and not wrapped a second time. The answer is cached in a module-level `weakref.WeakKeyDictionary` keyed on the `F` object, so repeated calls on one model, as `EventLoop` makes per segment, inspect it once; an `F` that cannot be weakly referenced is inspected at every call. If `alg.explicit`, raise `TypeError` unless `M` pairs rows and variables one to one (Section 12.1).
6. `stats = Stats(alg.scheme)`, `stats.ncondition = 0`; `linalg = IterationMatrix(self)`; `_daeic_backend = get_linsolver()`; `_pbar = tqdm(total=tend - t0)` when `opts.pbar`, as `rodas.py:88-89`.
7. Inside `with linsolver(self._daeic_backend):`, `y = DaeIc(proxy, u0, t0, opts.rtol)` with `proxy` a `SimpleNamespace(M=M, p=p, F=counted_F, J=counted_J)`, so its calls are counted (Section 5.9). At initialization the block changes nothing, since it sets the backend that is already global; it matters for the later `DaeIc` calls of Section 11.7, which then use the same backend as this one even when `integ.solve()` runs outside the caller's `with linsolver(...)` block. `DaeIc` itself is unchanged and keeps using the global backend without a cache (`daeic.py:44, :55`), as legacy Rodas does (`rodas.py:85`). The guard around `DaeIc` catches exactly four failures: `ValueError` whose message is `'Need Better y0'` (`daeic.py:68`), `np.linalg.LinAlgError` and `RuntimeError` from a singular algebraic Jacobian in its solves (`daeic.py:44, :55`, `laesolver.py:105-125`), and `StepFailure` from an arithmetic error of `F` or `J` (Section 5.8). Any of these makes the run fail at `t0` (Section 5.8) with `T = [t0]` and `Y = [u0]`, and the message quotes the exception. Any other `ValueError`, for example a NumPy broadcast error or the below-range error of a `TimeSeriesParam` whose first stamp lies after `t0` (`Solverz/equation/param.py:185-212`), is a programming error and propagates. A `RuntimeError` raised by the user's `F` inside `DaeIc` cannot be told apart from a solver failure and is reported as a failure with its message.
8. `u = y` copied into the owned buffer, `uprev = u.copy()`, `u_step = u`, `t_step = t0`, `sol.push(t0, u.copy())`.
9. `controller = alg.controller(opts)`, `cache = alg.alloc(self)`.
10. Default configuration: `tstops` = the heap of the distinct values of `tstops` and of the callbacks' `tstops` in `(t0, tend)`, plus `tend` if `tend > t0`. Legacy-compatible configuration: no heap.
11. Callbacks: evaluate every continuous condition at `(t0, u)` as its bottom values (Section 11.3). At most one `ContinuousCallback` may have `record=True`; otherwise `ValueError`.
12. `dt = policy.initial_dt(self, alg.initial_dt(self))`, `dtpropose = dt`, `new_step = True`, `iter = 0`, `accept_step = False`, `force_stepfail = False`, `nconsecutive_reject = 0`, `_nl_eta = 1.0`, `_interp_valid = True`, `terminated = failed = False`.
13. `finished = policy.at_end(self)`, which is true at once when `t0 == tend`. Such a run returns one row with `ret = 'success'`. Legacy fails there with "step size too small" (`rodas.py:135-138`); the degenerate case is outside the parity scope.

Recorded at I3. `Integrator.__init__(dae, tspan, y0, alg, opt=None)` of I3 performs the steps that one attempt needs, namely step 1, step 2 without the `tstops` check, the copy of step 4, step 5 without the `explicit` check, step 6 without `_daeic_backend` and the progress bar, the buffers of step 8 without saving, the `alloc` of step 9, and the fields of step 12 that the dispatch reads, with `dt = None`; I4 adds `callbacks`, `tstops` and the other steps. The adapter of step 5 is written in `integrator.py` with the body of `_with_out` instead of calling `_with_out`, since `_with_out` reads the signature with `follow_wrapped=True` and would return unchanged a `functools.wraps` wrapper without `out` around a residual that has one.

Recorded at I4. `from_opt` already applies the `tspan` rule of step 3 (Section 4.2), so the Integrator reads `t0` and `tend` from `opts` and the policies have no `prepare_tspan`. `DefaultPolicy` gains at I4 only the two members the initialization reads, `initial_dt` and `at_end`, since the tests of I3 build Integrators in the default configuration; its loop members follow with I5, and until then only the legacy-compatible configuration runs the loop. Until I6a, `Integrator.__init__` raises `NotImplementedError` for `opt.event` and a non-empty `callbacks`, after the `tstops` check of step 2, instead of ignoring them. When `DaeIc` fails, the initialization still completes steps 8 to 12, so that the Integrator is whole, and then fails the run. The progress bar of step 6 is created after `DaeIc`, where legacy creates it (`rodas.py:85, :88-89`), so that an exception that `DaeIc` propagates leaves no open bar.

Recorded at I6a. `Integrator.__init__` accepts `opt.event` and continuous callbacks without an `affect`; until I6b it raises `NotImplementedError` for a `ContinuousCallback` with an `affect`, and `TypeError` for an entry of `callbacks` that is not a `ContinuousCallback`, after the `tstops` check of step 2. `__all__` gains `ContinuousCallback`; `DiscreteCallback` and `preset_time_callback` follow with I6b.

Recorded at I6b. `Integrator.__init__` accepts a `ContinuousCallback` with an `affect` and `DiscreteCallback` objects, whose `tstops` join the heap of step 10, and raises `TypeError` for any other entry of `callbacks`. `__all__` gains `DiscreteCallback` and `preset_time_callback`.

Recorded after I8. A sparse algebraic Jacobian that is exactly singular raises no `RuntimeError` in `DaeIc`: `laesolver.solve` catches the failure of KLU and falls back to `spsolve` (`laesolver.py:112-123`), which warns and returns `NaN`, and `DaeIc` can then return a state with `NaN` entries, since its convergence test measures the correction only on the entries that are not `NaN` and passes when none is left (`daeic.py:45-46, :56, :65-66`). The guard therefore also fails the run when the state `DaeIc` returns is not finite, with the reason `it returned a state that is not finite`, instead of starting the run from `NaN`, where it would fail later at the rejection limit for a reason that does not name the cause.

### 5.3 The loop

The names follow SciML (`ODE@7393799:lib/OrdinaryDiffEqCore/src/integrators/integrator_utils.jl:84-127, :175-203, :600-680`), so that behaviour can be compared line by line.

```python
def step(self):
    if self.finished:
        return False
    while True:
        if not self.loopheader():                 # commit, check_error, bounds, tstops
            return False                          # run failed, message printed once
        if self.next_step_tstop and abs(self.dt) < math.ulp(abs(self.t)):
            self._skip_to_tstop()                 # default configuration only
        else:
            self.perform_step()                   # the algorithm; StepFailure -> force_stepfail
        self.loopfooter()                         # accept or reject, t, callbacks, saving
        if self.failed:
            return False
        if self.accept_step:
            self.policy.handle_tstop(self)        # pop every tstop equal to t
            self.finished = self.terminated or self.policy.at_end(self)
            return not self.finished

def solve(self):
    while self.step():
        pass
    return self.postamble()

def terminate(self):
    self.terminated = True
    self.retcode = 'terminated'
    self.finished = True

def loopheader(self):
    if self.iter > 0:
        if self.accept_step:
            self.apply_step()                     # COMMIT POINT
        elif not self.force_stepfail:
            self.controller.on_reject(self, self.q)   # sets self.dt
        # after a failed attempt loopfooter has already divided dt by failfactor
    reason = self.policy.check_error(self)        # on the dt proposed for this attempt
    if reason is not None:
        self._fail(reason)
        return False
    self.iter += 1
    self.policy.fix_dt_at_bounds(self)
    self.policy.modify_dt_for_tstops(self)
    self.force_stepfail = False
    return True

def apply_step(self):
    np.copyto(self.uprev, self.u)                 # uprev <- u: the accepted state is committed here
    self.u_step = self.u                          # re-attach the step's end state to u
    self._interp_valid = True
    self.new_step = True
    self.dt = self.dtpropose

def loopfooter(self):
    if self.force_stepfail:
        self.accept_step = False
        self.stats.nreject += 1
        self.nconsecutive_reject += 1
        if self.opts.adaptive:
            self.dt = self.dt / self.opts.failfactor
        else:
            self._fail('the step failed at a fixed step size: ' + self._stepfail_reason)
        return
    if self.opts.adaptive:
        self.policy.sanitize_EEst(self)
        self.q = self.controller.stepsize(self)
        self.accept_step = self.controller.accepts(self)
    elif not self._all_finite(self.u):            # np.isfinite(u, out=fin); fin.all()
        self._fail('the state is not finite')
        return
    else:
        self.accept_step = True
    if not self.accept_step:
        self.stats.nreject += 1
        self.nconsecutive_reject += 1
        return
    self.stats.nstep += 1
    self.nconsecutive_reject = 0
    self.dt_step = self.dt
    ttmp = self.t + self.dt
    self.tprev = self.t
    self.t = self.tstop_target if self.next_step_tstop else ttmp
    self.t_step = self.t
    if self.opts.adaptive:
        self.dtpropose = self.policy.dt_propose(self, self.controller.on_accept(self, self.q))
    else:
        self.dtpropose = self.opts.dt0
    self.next_step_tstop = False
    self.handle_callbacks()                       # Section 11; saving happens here
    if self.opts.pbar:
        self._pbar.update(self.t - self.tprev)
```

`terminate()` may be called by an `affect` or by the caller between two `step()` calls; in both cases the next `step()` returns `False` and `solve()` goes to `postamble`.

`perform_step()` is the dispatch of Section 12.1. It clears `_W_cache`; when `new_step` is true it invalidates `_J0`, `_F0` and `_ft`; it calls the algorithm in its style; it catches `StepFailure` and `ArithmeticError` into `force_stepfail = True` and `_stepfail_reason` (Section 5.8); it sets `_interp_ready = False` and, at the end, `new_step = False`. `_skip_to_tstop()` copies `uprev` into `u`, sets `EEst = 0.0` and marks the attempt so that `loopfooter` accepts it without calling the controller and sets `dtpropose = dt_untruncated`; this is SciML's `handle_tstop_step!` (`integrator_utils.jl:329-336`).

### 5.4 The commit point and the step's end state

The accepted state is committed in `apply_step`, at the top of the loopheader that follows the accepted attempt, and nowhere else. Between `loopfooter` and that commit, `(tprev, uprev)` is the start and `(t_step, u_step)` the end of the accepted step, `dt_step` its length, and the algorithm's cache still describes it. Callbacks, saving and `Integrator.interp` run in this window. `step()` returns inside it, so a caller can inspect `t`, `u` and interpolate before the next `step()`. `perform_step` reads `t`, `dt` and `uprev`, writes `u` and `EEst`, and never changes `t`, `tprev` or `uprev`. After a rejected attempt `u` holds a rejected trial state; it is never saved, and failure never reads it.

`u_step` is the same object as `u` after acceptance, so that no copy is made on an ordinary step. The core never writes into `u_step`. Before the core itself changes `u` after acceptance, namely when an acting event moves the state to `te` (Section 11.5), before any `affect` runs, and before `DaeIc` in the modification protocol, it detaches the end state: `np.copyto(self._u_step_buf, self.u); self.u_step = self._u_step_buf`, done at most once per step. Every interpolant reads the end of the step from `u_step` and `t_step` (Sections 10.5 and 12.1), so the saved nodes before an event and the root finding of other callbacks see the step that was actually computed, whatever has happened to `u`. `apply_step` re-attaches `u_step = u`. A caller who changes `integ.u` between two `step()` calls cannot trigger the detach, so `model_modified()` marks the step's interpolant invalid (`_interp_valid = False`), and `interp` then accepts only `tq == t` until the next accepted step.

### 5.5 `tstops`, `dtpropose` and bounds

Default configuration:

- `fix_dt_at_bounds`: `dt = min(dt, dtmax)`, then `dt = max(dt, dtmin(t))` with `dtmin(t) = 16 * math.ulp(abs(t))`. The constant 16 is the legacy one (`rodas.py:80`), taken relative to the current `t` instead of `t0`. `math.ulp` equals `np.spacing` for every finite non-negative argument and is a plain C call, while the NumPy function passes through NumPy's scalar machinery on every attempt; the same holds for `math.isfinite` against `np.isfinite` on a scalar. `DefaultPolicy` uses the `math` functions for scalars, and `LegacyRodasPolicy` keeps legacy's NumPy expressions.
- `modify_dt_for_tstops` (`integrator_utils.jl:271-327`, adaptive branch): with `tstop = tstops[0]`, `distance = tstop - t` and `tol = 100 * math.ulp(max(abs(t), abs(tstop)))`, store `dt_untruncated = dt`; if `dt + tol < distance` the attempt is not a tstop attempt; otherwise set `next_step_tstop = True`, `tstop_target = tstop` and `dt = distance`. On acceptance `t = tstop_target` exactly. Setting `dt = distance`, also when `dt` lay up to `tol` below it, makes the step computed by the algorithm end at the stop time up to one rounding of `tprev + dt_step`, so `t` and `tprev + dt_step` differ by at most one unit in the last place. SciML keeps `dt = min(dt, distance)` and so leaves a gap of up to 100 units between the computed step and the time it is assigned.
- `dt_propose(dtnew)`: if the accepted attempt was a tstop attempt with `dt_untruncated > dt`, then `dtnew = max(dtnew, dt_untruncated)`. This is the intent of design rule 2 (plan:87): a step shortened only to meet a stop time does not shrink the next one. SciML's code runs `modify_dt_for_tstops!` twice on the accepted path and restores the already truncated value instead (`integrator_utils.jl:109, :122-123, :191`); the core runs it once. Then `dtpropose = min(dtmax, max(dtmin(t), dtnew))`.
- `handle_tstop`: pop every heap entry equal to `t`.
- Non-adaptive runs use `dt = dt0` for every step, truncated at `tstops` in the same way.

The legacy-compatible configuration has its own rules, listed in Section 6.

### 5.6 Saving

- `len(tspan) == 2`: every accepted step pushes `(t, u.copy())`. In the default configuration the push is skipped when the last saved time already equals `t`, which happens only after a callback saved there; left and right saves of callbacks (Section 11.5) always push.
- `len(tspan) > 2`, default configuration: `savevalues` pops every node `tq <= t`; if `tq == t` it pushes `u.copy()`, otherwise `interp(tq)` (`integrator_utils.jl:343-432`). Nodes never enter the `tstops` heap, so they never change the step sequence. `tend` is a node and the last `tstop`, so its row is `u` exactly.
- Legacy-compatible configuration: Section 6.
- `SolutionBuffer` keeps a Python list of times and a list of row copies, and `to_daesol` stacks them once. There is no capacity and no growth rule, unlike the legacy buffers of 10001 rows grown by 1000 (`rodas.py:82-84, :340-344`) and the fixed event buffers of 10001 rows (`rodas.py:106-108`). Recorded at I1: `EventLog` keeps one list per field in the same way; both keep the arrays they are given, so the caller passes copies, as Section 5.2, step 8 does, and the components found at one event time may share one state.

### 5.7 Termination

The run is finished when `terminated` or `failed` is set, or when the configuration's end test holds after an accepted step: in the default configuration the `tstops` heap is empty, which happens when `t == tend`; the legacy-compatible test is in Section 6. In the default configuration `T[-1] == tend` exactly after a successful run.

`postamble()` closes the progress bar; if the run was terminated and the last saved time differs from `t`, it pushes `(t, u.copy())`; it sets `retcode` to `'success'` unless it is already `'terminated'` or `'failed'`, copies it into `stats.ret` and sets `stats.succeed = retcode != 'failed'`; and it returns `to_daesol(...)`, which stacks the rows, builds `te`, `ye`, `ie` from the `EventLog` and applies the `Vars` conversion of Section 4.4. A normal run never needs the extra row, since its last accepted step is saved at `t` itself.

### 5.8 The failure path

A failure (`_fail(reason)`) sets `failed = True`, `finished = True`, `retcode = 'failed'`, `stats.ret = 'failed'`, `stats.succeed = False`, `stats.t_fail = t`, prints exactly one line, and returns the rows saved so far. It never raises. The printed line is `f"{alg.scheme}: {reason} at t = {t!r}; the solution is returned up to t = {sol.last_t!r}."`. The reasons are:

| Reason | Default configuration | Legacy-compatible configuration |
|---|---|---|
| step size not finite | `not math.isfinite(dt)` in `check_error` | `not np.isfinite(dt)` |
| too many rejections | `nconsecutive_reject > 100` | `reject > 100` (`rodas.py:140-143`), the same count |
| step size too small | the last attempt was rejected and `dt < dtmin(t)` before `fix_dt_at_bounds` | `np.abs(dt) < np.spacing(1.0)` before the stretch (`rodas.py:135-138`) |
| fixed step failed | `force_stepfail` in a non-adaptive run | same |
| state not finite | non-adaptive run with a non-finite `u` | same |
| inconsistent initial values | one of the `DaeIc` failures of Section 5.2, step 7, at `t0` or in a modification protocol | same |
| model error outside a step | a `StepFailure` from an arithmetic error of `F` or `J` in `addsteps` or in the interpolation of a callback | same |

Inside a step nothing raises (design rule 7, plan:92). `IterationMatrix` turns `RuntimeError` from KLU, SuperLU or `klu_solve`, and `np.linalg.LinAlgError` from the dense path, into `StepFailure`; `implicit()` raises `StepFailure` on a Newton failure; an algorithm may raise `StepFailure` itself. The counted services `F` and `J` turn an `ArithmeticError` (`ZeroDivisionError`, `FloatingPointError`, `OverflowError`) raised by `dae.F` or `dae.J` into `StepFailure(f"{type(e).__name__} in F: {e}")`. This matters for rendered models: they are compiled with `@njit(cache=True)` (`Solverz/code_printer/python/module/module_generator.py:402-447`) and so with Numba's default `error_model='python'`, which raises `ZeroDivisionError` on a scalar float division by zero, as in a `LoopEqn` kernel, where NumPy returns `inf`. With the conversion, a trial stage that divides by zero rejects the attempt in both forms, and `EventLoop` keeps its prefix and its `IntegrationFailure(t, y)` (`SolPSDyn@c16a041:SolPSDyn/dae/event_loop.py:458-475`). The dispatch in `perform_step` catches `StepFailure` and any `ArithmeticError` raised during the attempt, including one from the core's own NumPy arithmetic under `np.seterr(all='raise')`. Legacy Rodas stops silently on a factorization `RuntimeError` with `ret` unset (`rodas.py:186-189`), lets a `klu_solve` error escape (`rodas.py:191, :200`) and lets an arithmetic error escape; the core instead fails the attempt and retries with half the step, and D3b claims no parity for a run on which legacy raises. Exceptions of any other type propagate. Among them are a `TypeError` from a user's residual or callback, which is a programming error, and the `SystemError` and `TimeoutError` by which a private study script interrupts a run through `SIGALRM`, which must keep propagating. Non-finite values in an adaptive run are not failures but rejections: `DefaultPolicy.sanitize_EEst` turns them into `EEst = np.inf` (Section 10.4), the controller then shrinks the step by `qmin`, and the step-size or rejection limit ends the run if the values stay non-finite.

On failure `postamble` does not append the current state: the prefix is the list of rows saved by accepted steps. With an output grid that prefix holds nodes only, as in legacy, and `stats.t_fail` tells how far the run got. `EventLoop` then emits the prefix and raises `IntegrationFailure(t=T[-1], y=Y[-1])` (`SolPSDyn@c16a041:SolPSDyn/dae/event_loop.py:451-476`).

Recorded at I4. The printed times are `float(t)!r` and `float(sol.last_t)!r`, since under NumPy 2 the repr of an `np.float64`, the type of `t` in the legacy-compatible configuration, reads `np.float64(...)`. The model error outside a step is caught around `handle_callbacks` in `loopfooter`: the accepted step stands, the run fails at its end, and the reason reads `the model failed after the step was accepted: ` followed by the message of the `StepFailure`.

### 5.9 `Stats`

The core counts every call it makes.

| Field | Core | Legacy Rodas |
|---|---|---|
| `nstep` | accepted steps | accepted steps (`rodas.py:226`) |
| `nreject` | rejected attempts, including failed attempts | error-test rejections (`rodas.py:353`) |
| `nfeval` | every call of `dae.F`: `DaeIc` at start and in modification protocols, `F0` once per step, `dF/dt` once per step, stage residuals, Newton residuals, residuals for error estimates, `s.f`, and the end residual that the Rodas3 interpolant needs once per interpolated step | `1 + (s - 1)` per attempt (`rodas.py:164, :199`); the two calls in `dfdt` (`rodas.py:378-379`), the two calls per interpolated node of `ntrp2` (`rodas.py:404-405`) and the calls in `DaeIc` are not counted |
| `nJeval` | every call of `dae.J`, including those in `DaeIc` | once per step (`rodas.py:158-160`) |
| `ndecomp` | every factorization of `W` | once per attempt (`rodas.py:190`), although the dense path performs `s` LAPACK factorizations per attempt (`laesolver.py:156-162`) |
| `nsolve` | every solve with `W` | `s` per attempt (`rodas.py:202`) |
| `ncondition` | every call of a continuous or discrete condition; a vector condition counts once per call | none |

For a Rosenbrock step of `s` stages the core makes `s + 1` residual calls on the first attempt and `s - 1` on each retry, since `F(t, uprev)` and `dF/dt` are kept across retries. Legacy makes `s + 2` calls per attempt and counts `s`. Parity tests therefore compare states, never `Stats`. `EventLoop` folds these counters into its own statistics (`event_loop.py:681-689`), whose values change accordingly.

### 5.10 Model and parameters: reading and mutation

- The core reads `dae.M` and `dae.p` into `integ.M` and `integ.p` at initialization and again in every modification protocol. Within a step both are fixed.
- An in-place change to an array inside `dae.p` or to `dae.M.data` is visible at once, because the core holds the same objects. `ModeSwitch` changes `dae.M.data` in place and keeps the CSC pattern (`SolPSDyn@c16a041:SolPSDyn/dae/limiter.py:345-350, :402-404`), and it sorts the indices of that same object (`limiter.py:316-321`), so the core must never replace `dae.M` by a converted copy.
- A change of the model inside a call is supported in two places only: in a callback's `affect`, and between two `step()` calls followed by `model_modified()`. Both run the modification protocol, which increases `model_epoch`. A change made anywhere else within a call is unsupported, since caches derived from `M` would not see it. A rebinding of `dae.p` or `dae.M` between calls is picked up at the next call.
- Nothing derived from `M` is cached across calls. Within a call, caches derived from `M`, namely the vector `D` of algebraic rows (Section 12.2), the row pairing of the Rodas3 interpolant together with the paired values of `M` (Section 10.5), and the pairing of `s.f`, store the `model_epoch` they were built for and are rebuilt when it changes.
- Legacy passes the `p` captured at call entry to `J` and to the stage residuals (`rodas.py:120`) while `dfdt` and `ntrp2` read `dae.p` live (`rodas.py:378-379, :404-405`). The core reads `integ.p` everywhere. The two agree whenever `dae.p` is not rebound during a call, which covers every event-free run.
- `F` must be a pure function of `(t, y, p)`. The core evaluates `F(t, uprev)` once per step and reuses it across retries, where legacy evaluates it again at every attempt (`rodas.py:163`) and twice more in `dfdt`. The generated residuals are pure: a trigger is a local assignment computed from the current state (`Solverz/code_printer/python/utilities.py:119-130`), and `TimeSeriesParam.get_v_t` has no side effect (`Solverz/equation/param.py:187-212`).

## 6. The two configurations

The legacy-compatible configuration is selected by `Rodas3(legacy_compat=True)`, `Rodas4(legacy_compat=True)`, `Rodasp(legacy_compat=True)` or `Rodas5P(legacy_compat=True)`. It exists to reproduce legacy Rodas. `Integrator.__init__` builds `LegacyRodasPolicy` when `alg.legacy_compat` is true and `DefaultPolicy` otherwise; the loop of Section 5.3 contains no branch on the configuration.

| Aspect | `DefaultPolicy` | `LegacyRodasPolicy`, legacy reference |
|---|---|---|
| `tspan` | float64 copy; strictly increasing when `len > 2` | `np.array(tspan)` unconverted (`rodas.py:71`); a non-increasing grid is accepted, and saving then stops at the first repeated node as in legacy |
| initial `dt` | `alg.initial_dt`, then `min(dtmax, max(dtmin(t0), dt))` | `dt0` or `1e-6 * (tend - t0)`, then `dt = np.maximum(dt, hmin)`, `dt = np.minimum(dt, hmax)` (`rodas.py:111-117`) |
| `dtmax` | `hmax` or `np.abs(tend - t0)` | same value; never written to `opt` (`rodas.py:76-77`) |
| smallest step | `dtmin(t) = 16 * math.ulp(abs(t))` | `hmin = 16 * np.spacing(t0)`, fixed for the call (`rodas.py:80`) |
| `check_error` | Section 5.8 | `np.abs(dt) < np.spacing(1.0)`, then `reject > 100` (`rodas.py:135-143`), then `not np.isfinite(dt)` |
| `fix_dt_at_bounds` | `min(dtmax, ...)`, `max(dtmin(t), ...)` | `if t + dt >= tend: dt = tend - t` `else: dt = np.minimum(dt, 0.5 * (tend - t))` (`rodas.py:146-149`); a non-adaptive run sets `dt = dt0` first and applies the first branch only |
| `tstops` | heap with `tend`, exact landing | refused with `ValueError` |
| new `t` | `tstop_target` or `t + dt` | `t + dt` (`rodas.py:224`) |
| end test | heap empty | `np.abs(tend - t) < np.spacing(1.0)` after an accepted step (`rodas.py:346`) |
| finished | `terminated or failed or` end test | `terminated or failed or` end test, which is legacy's `... or stop` (`rodas.py:346`) |
| controller | `IController` | `LegacyRodasController` (Section 7.3) |
| `sanitize_EEst` | `np.inf` unless `EEst` and `u` are finite | identity (Section 10.4) |
| `dF/dt` | `'ode23s'` | `'legacy'` (Section 9) |
| error norm | Section 10.4, default | Section 10.4, legacy (`rodas.py:207-212`) |
| KLU and SuperLU cache | per-call `KLUCache`; a KLU analysis without matching shared through `model_cache(dae)` (Section 8.4) | fresh `KLUCache` per call (`rodas.py:124`) |
| dense `J` | one LAPACK factorization per `W` | `np.linalg.solve` at every stage solve (`laesolver.py:156-162`) |
| saving, `len > 2` | Section 5.6 | the legacy loop: once per accepted step, before its first node, `alg.addsteps(integ, cache)`; then `while idx < len(tspan) and t >= tspan[idx] > tprev:` push `(tspan[idx], alg.interpolant(theta))` with `theta = (tspan[idx] - tprev) / dt_step` (`rodas.py:301-331`); every node, `tend` included, comes from the interpolant. A node equal to the time `te` of an event handled in the step is saved as `u`, so that `Y[-1]` is byte-equal to `ye` in this configuration too; an event-free run never takes this branch |
| saving, `len == 2` | every accepted step | every accepted step (`rodas.py:332-335`) |
| Rodas3 interpolant | `ntrp2` with slopes paired through `M` (Section 10.5) | `ntrp2` with raw residuals (`rodas.py:401-407`) |
| `NaN` warning print | none | none; legacy prints `'Warning Rodas: NaN or Inf occurs.'` (`rodas.py:212`) |
| factorization failure | halve the step | halve the step; legacy stops silently (`rodas.py:186-189`) |

The legacy-compatible configuration accepts `opt.event` and continuous callbacks, which then follow the event semantics of Section 11; a terminal event stops the run as in the default configuration. D3b claims parity only on runs where all of the following hold: no event and no callback; `opt.fix_h` false, since the legacy fixed-step form can overshoot `tend` and never terminate (`rodas.py:151-152, :346`); `hinit` either `None` or positive; `t0 < tend`; the class that matches `opt.scheme`; and a legacy run that neither raises nor stops on a factorization error. Parity also holds only per call for the `Opt` state that legacy sees at entry: legacy writes `hmax` and `facmax` into a shared `Opt` (`rodas.py:77, :349, :354`), so a parity test gives each legacy call its own `Opt`. The user guide states these conditions.

## 7. Controllers

### 7.1 Protocol

```python
class Controller:
    def __init__(self, opts, alg): ...
    def stepsize(self, integ): ...          # after every adaptive attempt that did not fail; returns q
    def accepts(self, integ) -> bool: ...
    def on_accept(self, integ, q): ...      # returns the new step
    def on_reject(self, integ, q): ...      # sets integ.dt
```

The controller's state lives in the controller object, which belongs to the Integrator. `Opt` is never written. An algorithm chooses its controller through `Algorithm.controller(opts)` (Section 12.1). `Algorithm.controller` returns `IController(opts, self)`; `Rosenbrock.controller` returns `LegacyRodasController(opts, self)` when `opts.legacy_compat`, else `IController(opts, self)`. An author who prefers PI control writes `def controller(self, opts): return PIController(opts, self)`.

Recorded at I4. The base `Controller.accepts` is `EEst <= 1.0`, which the three controllers share, and the other members of the base raise `NotImplementedError`. `IController` and `PIController` read `alg.error_order`, or `alg.order` when it is `None`, the default that Section 12.1 states.

### 7.2 `IController`, the default

Parameters: `safety = opts.safety` (0.9), `qmin = opts.qmin` (0.2), `qmax = opts.qmax` (6), `qmax_init = opts.qmax_init` (6), `qsteady_min = qsteady_max = 1.0`, `expo = 1 / alg.error_order`. State: `qmax_now = qmax_init`.

```python
def stepsize(self, integ):
    E = integ.EEst
    if E == 0:
        return 1.0 / self.qmax_now
    return max(1.0 / self.qmax_now, min(1.0 / self.qmin, E ** self.expo / self.safety))

def accepts(self, integ):
    return integ.EEst <= 1.0

def on_accept(self, integ, q):
    if self.qsteady_min <= q <= self.qsteady_max:
        q = 1.0
    self.qmax_now = self.qmax
    return integ.dt / q

def on_reject(self, integ, q):
    integ.dt = integ.dt / q
    self.qmax_now = 1.0
```

`q` is the inverse growth factor (`ODE@7393799:lib/OrdinaryDiffEqCore/src/integrators/controllers.jl:659-698`). For the Rosenbrock family `error_order = pord`, so `expo = 1 / pord`, the legacy exponent (`rodas.py:214`). After a rejection `qmax_now = 1`, so no attempt grows the step until the next acceptance, which is the legacy rule `facmax = 1` until the next accepted step (`rodas.py:349, :354`). The steady band `[1, 1]` has no effect other than at `q == 1`: strict Rosenbrock methods refactorize at every step anyway, and legacy has no band. The formula is SciML's in exact floating point; SciML itself evaluates the powers with a Float32 approximation (`controllers.jl:675-676`), so no bit comparison with SciML is possible or intended.

### 7.3 `LegacyRodasController`

Parameters from `opts` and the tableau: `safety`, `fac1 = opts.qmin`, `fac2 = opts.qmax`, `pord`, `hmin = 16 * np.spacing(opts.t0)` with `t0` in its `tspan` dtype, `hmax = opts.dtmax`, as in Section 6. State: `facmax = opts.qmax_init`, the value `opt.facmax` has at call entry, and `dtnew`.

```python
def stepsize(self, integ):                  # rodas.py:213-216
    err = np.maximum(integ.EEst, 1.0e-6)
    fac = self.safety / (err ** (1 / self.pord))
    fac = np.minimum(self.facmax, np.maximum(self.fac1, fac))
    self.dtnew = integ.dt * fac
    return fac

def accepts(self, integ):                    # rodas.py:221
    return integ.EEst <= 1.0

def on_accept(self, integ, q):               # rodas.py:349, :355
    self.facmax = self.fac2
    return np.min([self.hmax, np.max([self.hmin, self.dtnew])])

def on_reject(self, integ, q):               # rodas.py:354, :355
    self.facmax = 1
    integ.dt = np.min([self.hmax, np.max([self.hmin, self.dtnew])])
```

Every expression is the legacy one with the same operand order and the same NumPy functions. `integ.EEst` stays the `np.float64` returned by `np.max`; it is never converted to a Python `float`, so `err ** (1 / pord)` runs through the same scalar power as in legacy. `EEst` is the legacy error before the floor of `1e-6`; since the floor is below 1, `EEst <= 1.0` decides exactly as the floored value does, and a `NaN` rejects in both. `LegacyRodasPolicy.dt_propose` returns the value of `on_accept` unchanged.

### 7.4 `PIController`

Included because issue #189 names it for `controllers.py`; no C2 algorithm selects it. With `k = alg.error_order`, `beta1 = 7 / (10 * k)`, `beta2 = 2 / (5 * k)`, `qoldinit = 1e-4`, and state `q11 = 1.0`, `errold = qoldinit`, `qmax_now = qmax_init` (`controllers.jl:754-843`):

- `stepsize`: if `EEst == 0` return `1 / qmax_now`; else `q11 = EEst ** beta1`, `q = q11 / errold ** beta2`, return `max(1 / qmax_now, min(1 / qmin, q / safety))`.
- `accepts`: `EEst <= 1`.
- `on_accept`: steady band as `IController`; `errold = max(EEst, qoldinit)`; `qmax_now = qmax`; return `dt / q`.
- `on_reject`: `integ.dt = integ.dt / min(1 / qmin, q11 / safety)`; `qmax_now = 1.0`.

## 8. Linear algebra

### 8.1 Service

`integ.W(gamma)` returns the factorization of `W = M - (dt * gamma) * J0` for the current attempt, where `J0 = integ.J0()` is the Jacobian at `(t, uprev)`. `J0` is evaluated on the first request after `new_step` and reused on retries of the same step, which is legacy's `if reject == 0` (`rodas.py:158-160`) and design rule 6 (plan:91). Factorizations are kept in `_W_cache`, keyed by `gamma`, and cleared at the start of each attempt, so every attempt factorizes once per distinct `gamma`. The returned object has `rscale`, `dtgamma` and `solve(b, out=None)`.

Recorded at I2. `IterationMatrix.factorize(M, J, dt, gamma)` assembles and factorizes `W` and returns that object; `integ.W(gamma)` with its `_W_cache` belongs to the Integrator, which passes `integ.M`, `integ.J0()` and `integ.dt`. `ndecomp` and `nsolve` count the factorizations and solves that succeed, since a failed one raises `StepFailure`.

### 8.2 Build and solve, the legacy chain in both configurations

```python
dtgamma = integ.dt * gamma                               # rodas.py:173 evaluates dt * gamma first
Miter = integ.M - dtgamma * J0                           # rodas.py:173
row_max = np.max(np.abs(Miter), axis=1)                  # rodas.py:179
if self.rscale_to_dense is None:                         # decided once per call, rodas.py:180-181
    self.rscale_to_dense = hasattr(row_max, 'toarray')
if self.rscale_to_dense:
    row_max = row_max.toarray()                          # rodas.py:182-183
rscale = (1.0 / np.asarray(row_max)).ravel()             # rodas.py:184
Wm = diags_array(rscale, format='csc') @ Miter           # rodas.py:185
```

Both configurations use this chain, so the matrix handed to the factorization is the legacy one bit for bit, including three value-dependent effects. SciPy drops exact zeros in the subtraction and in the product, so the pattern of `W` can change from step to step (`docs/src/release_notes.md:25`). The product emits each column's row indices in descending order, which KLU receives unsorted and compares verbatim in its fingerprint (`Solverz/solvers/klu_backend.py:194-198, :307`). A row whose maximum is zero gives `rscale = inf`; the factorization or the solve then fails or returns non-finite values, which Sections 5.8 and 10.4 handle. Whether KLU's numeric factorization depends on the order of row indices within a column is not known, so an in-place assembly that emits sorted indices is not used.

The factorization is `lu_decomposition(Wm, backend=self.backend, cache=self.cache)` (`laesolver.py:128-153`) for a sparse `Wm` in both configurations and for a dense `Wm` in the legacy-compatible one, where it returns `dense_decomposition`, whose `solve` calls `np.linalg.solve` every time (`laesolver.py:156-162`). In the default configuration a dense `Wm` (an ndarray, or an `np.matrix` converted with `np.asarray`) is factorized once with `scipy.linalg.lapack.dgetrf`, and `solve` uses `dgetrs`; `info != 0` from either raises `StepFailure`. Apart from the cache lifetime of Section 8.4, this is the only point at which the default configuration factorizes differently from legacy, and it removes `s - 1` factorizations per Rosenbrock attempt on a dense model.

`solve(b, out=None)` first scales the right-hand side into a buffer that the factorization object owns, `np.multiply(self.rscale, b, out=self._scaled)`, which gives the bits of legacy's `rscale * rhs` (`rodas.py:191, :200`) without an allocation. It then solves:

- KLU: `lu.solve_into(self._scaled, out)`, a method added to `klu_decomposition` in `Solverz/solvers/klu_backend.py`. It performs the steps of the existing `solve` (`klu_backend.py:338-352`) into the caller's buffer: `np.take(b, perm, out=out)` when the analysis holds a matching, else `np.copyto(out, b)`, then `klu_solve` in place on `out`, and `RuntimeError` on a bad status. The values handed to `klu_solve` are those of `solve`, so the result is byte-equal. `out` must be a C-contiguous float64 vector of length `n` that does not share memory with `b`. The existing `solve` is unchanged, so no legacy caller is affected.
- Dense, default configuration: `dgetrs` on a copy of the scaled vector made into `out`, with `overwrite_b=True`.
- SuperLU and the dense legacy-compatible path: the backend returns a new array (`laesolver.py:274-277`, `:156-162`), which is copied into `out`. C2 accepts this allocation.

Without `out`, `solve` returns a new array. With `out`, it writes into `out` and returns it; `out` must be C-contiguous and of length `n`, otherwise `ValueError`, since a strided target such as a column of `K` cannot be handed to `klu_solve`.

Recorded at I2. The scaled right-hand side lives in one buffer of the `IterationMatrix` that all its factorizations share, since the backend consumes it before a solve returns, so no factorization allocates one, and `out` may be `b` itself. `solve` also raises `ValueError` for a `b` that is not a vector of length `n`, which `np.multiply` would broadcast silently, and `solve_into` checks both of its vectors, since `klu_solve` writes `n` values through the raw pointer of `out`.

### 8.3 Sparse and dense `J` and `M`

`J` is a `csc_array` for a sparse inline model (`Solverz/sym_algebra/symbols.py:366-367`) and for a rendered model (`Solverz/num_api/custom_function.py:264-274`), and a fresh ndarray for `made_numerical(..., sparse=False)`, the default (`Solverz/code_printer/python/inline/inline_printer.py:159`). `module_printer` has no dense option (`Solverz/code_printer/make_module.py:8-26`), so a rendered model always has a sparse `J`. `M` is a `csc_array` for every model built from a `DAE` (`Solverz/equation/equations.py:938`), and may be an ndarray in a hand-built `nDAE`. A sparse `M` minus a dense matrix is dense through SciPy's `_sub_dense` (`scipy/sparse/_base.py:718-719`, SciPy 1.16). The chain of Section 8.2 handles all four combinations, as legacy does (`docs/src/release_notes.md:102`).

### 8.4 Backend and cache lifetime

The backend is resolved once per call in `from_opt` and passed explicitly to every factorization. `init()` resolves it at initialization, so a later `integ.solve()` outside a `with linsolver(...)` block keeps the backend of `init()`.

- Legacy-compatible: `self.cache = KLUCache()` per Integrator, as `rodas.py:124`. The KLU symbolic analysis and the SuperLU ordering then start empty in every call, which D3b requires.
- Default: `self.cache = KLUCache()` per Integrator. At initialization the call takes `sym = model_cache(dae).symbolic` into `self.cache.symbolic` only if `sym is not None`, `sym.perm is None`, and the current setting would not compute a matching for this size, that is `not (klu_matching_enabled() and n >= klu_backend.MATCHING_MIN_N)`. After every factorization whose analysis holds no matching (`perm is None`), `model_cache(dae).symbolic` is set to it. An analysis without matching depends only on the pattern (`klu_backend.py:306-331`), so taking it from an earlier call gives the bits of a fresh analysis. An analysis with a matching is never shared, and each call computes its own at its first factorization and reuses it within the call, as legacy does. The SuperLU ordering always starts empty in every call (F1). A call's result therefore does not depend on earlier calls, on either backend and at any size.
- The cache has one slot. A pattern change, for example from dropped zeros, triggers a new analysis that replaces the slot (`klu_backend.py:307`). A KLU failure falls back to SuperLU for that factorization and leaves `cache.symbolic` unchanged (`laesolver.py:146-151`). `set_klu_matching` takes effect at the next analysis; within a call a cached analysis keeps the setting it was built with (`klu_backend.py:67-75`).
- The modification protocol of Section 11.7 resets the call's cache to the state from which a new call would start: `cache.superlu = None`, and `cache.symbolic = None` when it holds a matching. A run with an affect at `t1` is then byte-equal to two calls split at `t1`, on either backend, which the history check of Section 13 relies on. The cost is one COLAMD ordering under SuperLU, and one analysis above the matching threshold, per modification.
- Limitation. The pattern of `W` depends on its values. `ModeSwitch` keeps explicit zeros in `M.data` so that the pattern of `M` survives a mode change (`SolPSDyn@c16a041:SolPSDyn/dae/limiter.py:345-350, :402-404`), but a pinned switched row multiplies its free-rate Jacobian entries by `(1 - zu - zl) = 0` (`limiter.py:189-208`), and the chain of Section 8.2 then drops those exact zeros. Every engage or release therefore changes the pattern of `W` and costs a new analysis, as in legacy. Cross-call reuse saves an analysis only when consecutive calls start with the same pattern of `W`; I10 counts the analyses per run (Section 17.2). Building `W` on the fixed union pattern of `M` and `J` with explicit zeros kept would give one analysis per model; it changes the matrix handed to KLU and so the last bits relative to legacy, and it is recorded for a later milestone, for the default configuration only.

### 8.5 `J` is never mutated

The chain creates new matrices and passes only `Wm` to the backends. `sp_decomposition` sorts its input in place (`laesolver.py:243-244`), which is safe for `Wm`. `J` is never passed to `eliminate_zeros`, `sort_indices`, `sum_duplicates` or a backend. This matters because every `J` of a rendered model shares one `indices` and one `indptr` array (`custom_function.py:274`), and because `NeuralComposition` mutates the `J` it receives (`SolPSDyn@c16a041:SolPSDyn/neural/compose.py:49-60`).

## 9. `F0` and `dF/dt`

`integ.F0()` evaluates `F(t_start, uprev)` into the Integrator's buffer `_F0` at most once per step, on the first request after `new_step`, and returns a read-only view of the buffer. `t_start` is the start of the step: `integ.t` during an attempt and `integ.tprev` after acceptance. The buffer keeps its value until the next commit, so the Rosenbrock stage 0, `dF/dt`, the Rodas3 interpolant and formula algorithms share one evaluation.

`integ.dFdt(out=None)` returns the partial derivative of `F` with respect to `t` at `(t, uprev)`. It is computed once per step, on the first request after `new_step`, into the Integrator's buffer `_ft`, from `f0 = integ.F0()`, and reused on every retry. Both policies use a forward difference, since `TimeSeriesParam` cannot be evaluated before its first time stamp (`Solverz/equation/param.py:187-212`).

`'ode23s'`, the default, with `SQRT_EPS = np.sqrt(np.spacing(1.0))` (exactly `2**-26`) and `dt` the step of the first attempt of the step, after bounds and truncation:

```python
delt = SQRT_EPS * max(abs(t), abs(t + dt))
tdel = (t + min(delt, abs(dt))) - t
if tdel == 0.0:
    ft.fill(0.0)
else:
    F(t + tdel, uprev, out=scratch)
    np.subtract(scratch, f0, out=ft)
    np.divide(ft, tdel, out=ft)
```

`'legacy'`, the expressions of `rodas.py:376-380`:

```python
tscale = np.maximum(0.1 * np.abs(t), 1e-8)
ddt = t + np.sqrt(np.spacing(1)) * tscale - t
F(t + ddt, uprev, out=scratch)
np.subtract(scratch, f0, out=ft)
np.divide(ft, ddt, out=ft)
```

With `'legacy'` at `t = 0` the difference quotient uses `ddt = 1.49e-16`, which gave a 36 percent error in `dF/dt` on the SDCIB model (a private benchmark record). `'ode23s'` scales the increment with the step. The rounding error of a forward difference is about `eps` times the magnitude of the terms of `F` divided by the increment, and the increment is at least `SQRT_EPS * |dt|`, so the product `dt * ft` that enters the stages carries an error of about `SQRT_EPS` times the magnitude of the terms of `F`, at every `t` and every `dt`. The increment never exceeds `|dt|`, so the probe never leaves the step, and `TimeSeriesParam` returns its last value past its final stamp, so no probe fails at the end of a profile.

Caching rule for parity. Legacy recomputes `dfdt0 = dt * dfdt(dae, t, y0)` at every attempt (`rodas.py:162`), but `t` and `y0` do not change between attempts, and `dt` enters only through the final product. The core caches the unscaled `ft` and forms `dfdt0 = dt * ft` at every attempt; `g[j] * dfdt0` is formed afterwards, never folded into one scalar. `f0` is `F(t, uprev)`, which equals legacy's internal `f0` bit for bit under the purity rule of Section 5.10. The policy is fixed by the configuration and is not a separate public option.

Recorded at I2. `derivative.py` holds the two policies in `DFDT_POLICIES` as `dfdt_ode23s(F, t, dt, y, f0, ft, scratch)` and `dfdt_legacy` with the same signature, where `F` is the counted residual service `F(t, y, out)`; both write into `ft` and return it.

Recorded at I3. The dispatch of Section 5.3 records `t` as the start of the step when it invalidates `_J0`, `_F0` and `_ft`, and `J0`, `F0` and `dFdt` evaluate at that time, so that after acceptance they still describe `tprev`. `Rosenbrock.perform_step` requests `J0` through `W`, after `F0` and `dFdt` rather than first as step R0 lists it; the order changes no value, since `F` and `J` are pure functions of `(t, y, p)`.

## 10. The Rosenbrock family

### 10.1 Tables

A Rosenbrock method is its tableau: an object with the attributes of `Rodas_param` (`param.py:4-202`), namely `s`, `pord`, `gamma`, `alpha` and `gammatilde` (stored transposed, so `alpha[:, j]` is row `j` of the table), `a`, `g`, `b`, `bd`, `c`, `d`, `e`. The built-in classes use `Rodas_param` itself, so no coefficient is retyped; retyping would already change `rodas4`, whose `alpha[5, 1]` and `beta[4, 1]` differ in the last digits (`param.py:47, :53`).

```python
class Rodas3(Rosenbrock):  scheme = 'rodas3';  tableau = Rodas_param('rodas3');  interpolation = 'hermite'; interp_order = 2
class Rodas4(Rosenbrock):  scheme = 'rodas4';  tableau = Rodas_param('rodas4');  interpolation = 'ntrp1';   interp_order = 3
class Rodasp(Rosenbrock):  scheme = 'rodasp';  tableau = Rodas_param('rodasp');  interpolation = 'ntrp1';   interp_order = 3
class Rodas5P(Rosenbrock): scheme = 'rodas5p'; tableau = Rodas_param('rodas5p'); interpolation = 'ntrp1';   interp_order = 3
```

`Rosenbrock.__init__(self, *, legacy_compat=False)`; the argument is keyword-only, so that `solver=Rodas4` passed without parentheses, which calls `Rodas4(dae, tspan, y0, opt)`, fails with a `TypeError` instead of selecting a configuration. `Rosenbrock` sets `inplace = True`, `order = tableau.pord`, `error_order = tableau.pord`, `adaptive = True`, `norm = 'max'`. `interpolation` resolves in this order: an explicit class attribute wins; otherwise `'ntrp1'` if the tableau has `c`, `d` and `e`; otherwise the linear interpolant of `Algorithm`. The dense output `'ntrp1'` is `y0 + theta*h*K@(b + (theta - 1)*(c + theta*(d + theta*e)))`, evaluated by the compiled legacy function as `((theta*h)*K) @ v` (`rodas.py:411-414`). The values of `interp_order` are lower bounds: the dense outputs of RODAS and its successors are of order 3 or more, and the Rodas3 Hermite interpolant is linear for algebraic variables (Section 10.5). The kit of Section 13 measures them; a measured order below a declared value is reported to the lead, not hidden by lowering the value. `'rodas3d'` is not provided: `Rodas_param` defines no `c`, `d`, `e` for it (`param.py:171-200`), so dense output and events would fail; legacy Rodas keeps accepting it.

### 10.2 Cache

`Rosenbrock.alloc(integ)` allocates once: `K = np.zeros((n, s))` C-contiguous float64; the length-`n` float64 buffers `Fs`, `dfdt0`, `rhs`, `tmp`, `sum1`, `sum2`, `y1`, `utilde` and `x`, the contiguous target of the solves; the length-`s` buffers `dtb`, `dtbd`; the lists of views `alpha_cols[j] = tab.alpha[:, j]` and `gt_cols[j] = tab.gammatilde[:, j]`; and, for Rodas3, the length-`n` buffers `F1` for `F(t_step, u_step)` and `s0`, `s1` for the slopes of Section 10.5, with the `model_epoch` of the cached row pairing. `F(t, uprev)` lives in the Integrator's `_F0` (Section 9).

### 10.3 `perform_step`, in place, with the operation order of legacy

Notation: `t = integ.t`, `dt = integ.dt`, `y0 = integ.uprev`, `M = integ.M`, `tab = self.tableau`, `c = cache`. Every line keeps the operand order and the association of the cited legacy line. Multiplication and addition of two floating-point operands are commutative in IEEE arithmetic, so `np.multiply(x, dt)` equals `dt * x` bit for bit; the association of three or more operands is what must match, and it does.

| Step | Code | Legacy |
|---|---|---|
| R0 | `J0 = integ.J0()`; `F0 = integ.F0()`; `ft = integ.dFdt()`; each is evaluated on the first attempt of the step and reused on retries | `rodas.py:158-160, :162-163, :375-380` |
| R1 | `c.K.fill(0.0)` | `K = np.zeros((vsize, s))`, `rodas.py:156` |
| R2 | `np.multiply(ft, dt, out=c.dfdt0)` | `dfdt0 = dt * dfdt(...)`, `:162` |
| R3 | `np.multiply(c.dfdt0, tab.g[0], out=c.tmp)`; `np.add(F0, c.tmp, out=c.rhs)` | `rhs = F + g[0] * dfdt0`, `:163`; stage 0 has no `M @ sum_2` term |
| R4 | `W = integ.W(tab.gamma)` | `:173-187` via Section 8.2 |
| R5 | `W.solve(c.rhs, out=c.x)`; `c.K[:, 0] = c.x` | `K[:, 0] = lu.solve(rscale * rhs)`, `:191` |
| R6a | for `j = 1 .. s - 1`: `np.matmul(c.K, c.alpha_cols[j], out=c.sum1)` | `sum_1 = K @ alpha[:, j]`, `:194` |
| R6b | `np.matmul(c.K, c.gt_cols[j], out=c.sum2)` | `sum_2 = K @ gammatilde[:, j]`, `:195` |
| R6c | `np.multiply(c.sum1, dt, out=c.y1)`; `np.add(y0, c.y1, out=c.y1)` | `y1 = y0 + dt * sum_1`, `:196` |
| R6d | `integ.F(t + dt * tab.a[j], c.y1, out=c.Fs)` | `dae.F(t + dt * a[j], y1, p)`, `:198` |
| R6e | `Msum2 = M @ c.sum2`; `np.add(c.Fs, Msum2, out=c.rhs)` | `F + M @ sum_2`, `:198` |
| R6f | `np.multiply(c.dfdt0, tab.g[j], out=c.tmp)`; `np.add(c.rhs, c.tmp, out=c.rhs)` | `(...) + g[j] * dfdt0`, `:198` |
| R6g | `W.solve(c.rhs, out=c.x)`; `np.subtract(c.x, c.sum2, out=c.K[:, j])` | `K[:, j] = lu.solve(rscale * rhs) - sum_2`, `:200-201` |
| R7 | `np.multiply(tab.b, dt, out=c.dtb)`; `np.matmul(c.K, c.dtb, out=c.sum1)`; `np.add(y0, c.sum1, out=integ.u)` | `sum_1 = K @ (dt * b)`, `ynew = y0 + sum_1`, `:204-205` |
| R8 | if `integ.opts.adaptive`: `np.multiply(tab.bd, dt, out=c.dtbd)`; `np.matmul(c.K, c.dtbd, out=c.sum2)`; `np.subtract(c.sum1, c.sum2, out=c.utilde)`; `integ.EEst = integ.error_norm(c.utilde)` | `sum_2 = K @ (dt * bd)`, `err` from `sum_1 - sum_2`, `:206-212` |

Rules that the table implies:

- `K` stays `(n, s)` and C-contiguous, every stage product runs over all `s` columns including the zero ones, and `K` is zeroed at every attempt. A reused `K` with stale non-finite columns would turn `0 * inf` into `NaN`, and a transposed layout changes the BLAS call.
- The error vector is `K @ (dt*b) - K @ (dt*bd)` from two products, never `K @ (dt*(b - bd))`.
- No expression is fused into a multiply-add; the core uses no Numba `fastmath` and no `numexpr`.
- For a sparse `M`, `M @ sum2` is SciPy's matrix-vector product into a fresh zero vector (`scipy/sparse/_compressed.py:387-397`), which turns `-0.0` into `+0.0`; for an ndarray `M` it is NumPy's product. The core keeps the expression `M @ c.sum2` in both cases, including its allocation.
- `np.matmul(K, v, out=buf)` must equal `K @ v` bit for bit, also when `buf` holds `NaN` or `inf` from an earlier attempt, since `sum1`, `sum2` and `utilde` are reused and a BLAS that implements `beta = 0` by scaling would propagate `0 * NaN`. Milestone I1 checks this on the server for `n` in 2, 28, 1000 and 10000 and `s` in 4, 6 and 8, with `buf` prefilled with `NaN` and with `+inf`, before anything depends on it. If the check fails, the core writes `K @ v` and copies it with `np.copyto`, accepting one allocation per product. Recorded at I1: the check passed on the server, with NumPy 2.4.3 on OpenBLAS 0.3.31.dev, for every `K` of the stage loop, every column of `alpha` and `gammatilde`, and the vectors `dt * b` and `dt * bd`, so the core writes the products with `out=`.
- Remaining allocations per attempt: `J0` on the first attempt (the model's own); the SciPy chain of Section 8.2; one vector per `M @ sum2`; per solve, nothing under KLU and the default dense path, and the backend's result under SuperLU and the legacy-compatible dense path, plus the `b[perm]` gather inside SuperLU's permuted solve. Per interpolated node, `ntrp1` and `ntrp2` return a new array from Numba, which is copied into the row. Every residual is written through `out=`, and the error norms allocate nothing (Section 10.4).
- Non-adaptive runs skip R8 and never read `EEst`, as legacy's `fix_h` branch skips the error (`rodas.py:217-219`).
- `integ.dt_step` is set by the loop to the `dt` of the accepted attempt and is the step length of the interpolant.

### 10.4 Error norms

`integ.error_norm(utilde)` uses the Integrator's work buffers `w`, `w2` and a boolean buffer `fin`, and returns an `np.float64`.

Legacy norm, used by `LegacyRodasPolicy` for every algorithm (`rodas.py:208-211`):

```python
np.abs(u, out=w); np.multiply(w, rtol, out=w); np.add(w, atol, out=w)     # SK = atol + rtol*abs(ynew)
np.divide(utilde, w, out=w); np.abs(w, out=w); err = np.max(w)          # max |(sum_1 - sum_2)/SK|
np.isfinite(u, out=fin)
if not fin.all():                                                       # legacy: any isinf or any isnan
    err = 1.0e6
```

For float64, "some entry is infinite or `NaN`" is exactly "not every entry is finite", so the override fires in the same cases as legacy's two tests, without their two temporary arrays. The floor of `1e-6` belongs to `LegacyRodasController` (Section 7.3). `LegacyRodasPolicy.sanitize_EEst` is the identity, since the `1e6` override already lives inside this norm. A `NaN` error with a finite `u`, which `atol = 0` and a zero component produce as `0/0`, therefore stays `NaN`: it is rejected, and the controller makes the next step `NaN`. Legacy then keeps attempting with `dt = NaN` until its rejection limit (`rodas.py:213-221, :355`), while the core's `check_error` fails the run at the next attempt on the non-finite step. The saved prefix is the same.

Default norm, D4 and design rule 4:

```python
np.abs(u, out=w); np.abs(uprev, out=w2); np.maximum(w, w2, out=w)
np.multiply(w, rtol, out=w); np.add(w, atol, out=w)
np.divide(utilde, w, out=w)
E = np.max(np.abs(w, out=w))                    # alg.norm == 'max'
E = np.sqrt(np.mean(np.square(w, out=w)))       # alg.norm == 'rms', the default for new algorithms
```

`DefaultPolicy.sanitize_EEst` then sets `EEst = np.inf` unless both `EEst` and every entry of `u` are finite (`np.isfinite(u, out=fin)`, `fin.all()`). The `IController` turns `inf` into the smallest factor `qmin`, the same reduction by 0.2 that legacy reaches through `err = 1e6` and `fac1`.

Recorded at I3. `policies.py` exists from I3 on with the two members that one attempt reads: `dfdt`, the name of the configuration's policy in `DFDT_POLICIES`, and `error_norm(integ, e)`, the legacy norm in `LegacyRodasPolicy` and the default norm of `alg.norm` in `DefaultPolicy`, which raises `ValueError` for a `norm` other than `'rms'` and `'max'`. `integ.error_norm(e)` calls it, and the override returns `np.float64(1.0e6)`. I4 and I5 add the other members of the two policies.

### 10.5 Interpolation

`Rosenbrock.interpolant(integ, cache, theta, out)` with `theta = (tq - tprev) / dt_step` describes the last accepted step from `(tprev, uprev)` to `(t_step, u_step)` and never reads `integ.u`:

- `'ntrp1'` (Rodas4, Rodasp, Rodas5P, both configurations): `out[...] = ntrp1(integ.uprev, theta, integ.dt_step, cache.K, tab.b, tab.c, tab.d, tab.e)`, the compiled legacy function (`rodas.py:411-414`). Its product `((tau*dt)*K) @ v` runs through Numba's BLAS binding, not NumPy's (`numba/np/linalg.py:482-495`), so a NumPy re-implementation could differ in the last bits and is not used. `K` must be C-contiguous so that Numba uses the same specialization. At `theta = 1` the result is not bit-equal to `u_step`, since the association differs from R7; legacy saves this interpolated row at `tend` (`rodas.py:302-315`), and so does the legacy-compatible configuration.
- `'hermite'` (Rodas3): `out[...] = ntrp2(integ.uprev, integ.u_step, s0, s1, theta, integ.dt_step)` (`rodas.py:417-421`). `addsteps` computes the slope vectors once per step, before the first interpolation of that step, with the end residual `F1 = F(t_step, u_step)`, one counted residual per interpolated step. Legacy-compatible: `s0 = integ.F0()`, which is `F(tprev, uprev)`, and `s1 = F1`; legacy evaluates both at every interpolated node (`rodas.py:404-405`), with bit-equal results under Section 5.10, because `t_step` equals legacy's `told + dt` in this configuration. Default: legacy's `ntrp2` treats residual row `i` as the derivative of variable `i`, which is wrong when rows and variables are not aligned and gives algebraic variables the slope of an algebraic residual. The default therefore pairs them through `M`: with `(rows, cols) = M.nonzero()`, which drops explicit zeros, and only when no row and no column occurs twice, `s0[cols] = F0[rows] / Mv`, `s1[cols] = F1[rows] / Mv` with `Mv = M[rows, cols]`, and every other entry of `s0` and `s1` is the secant slope `(u_step - uprev) / dt_step`, which makes the Hermite polynomial linear in those entries. If some row or column occurs twice, every entry uses the secant slope. `rows`, `cols` and `Mv` are cached together per `model_epoch`, so the sparse fancy indexing runs once per model change.
- A table without `c`, `d`, `e` and without `interpolation = 'hermite'` uses the linear interpolant of `Algorithm`.

Recorded at I3. Both forms of the Rodas3 slopes are in `rosenbrock.py` from I3 on, since the cache of Section 10.2 carries the pairing; the milestone table lists the pairing with I5, whose tests exercise it through the loop. The pairing reads `rows`, `cols` and `Mv` from `M.tocoo()` with the entries equal to zero dropped, which for a sparse `M` is `M.nonzero()` with `M[rows, cols]` and needs no sparse fancy indexing; an entry stored twice then shows as a row that occurs twice, and every entry takes the secant slope. `interpolation` must be `'ntrp1'`, `'hermite'` or `'linear'`, and `'ntrp1'` needs a tableau with `c`, `d` and `e`; a subclass that breaks either rule raises `TypeError` when it is defined.

`integ.interp(tq, out=None)` checks `tprev <= tq <= t`, where `t` is the current time, which is `t_step` or, after an acting event, `te`. It returns a copy of `uprev` at `tq == tprev` and of `u` at `tq == t` exactly; after an `affect`, `u` at `t` is the modified state. Otherwise it calls `addsteps` once per step, then `interpolant`. `theta` may exceed 1 by one rounding, since `t` and `tprev + dt_step` can differ by one unit in the last place after a step that landed on a `tstop` (Section 5.5); every interpolant evaluates its polynomial there. After `model_modified()` only `tq == t` is accepted (Section 5.4). Outside the range `interp` raises `ValueError`, which is a programming error of the caller. The legacy-compatible saving calls `alg.addsteps` once per accepted step and then `alg.interpolant` directly for every node, `tend` included, as Section 6 requires.

### 10.6 Adding a Rosenbrock method by its table

```python
from Solverz.integrator import Rosenbrock, RosenbrockTableau

class MyRos(Rosenbrock):
    scheme = 'myros'
    interp_order = 1
    tableau = RosenbrockTableau.from_hairer(
        gamma=..., alpha=[[0, 0, 0], [..., 0, 0], [..., ..., 0]],
        beta=[[0, 0, 0], [..., 0, 0], [..., ..., 0]],
        b=[...], bd=[...], pord=3, c=None, d=None, e=None)
```

`from_hairer(gamma, alpha, *, beta=None, gamma_ij=None, b, bd, pord, c=None, d=None, e=None)` takes `alpha` as a strictly lower triangular `s x s` table in Hairer's notation and exactly one of `beta`, the strictly lower triangular table of `alpha_ij + gamma_ij`, or `gamma_ij`, the strictly lower triangular table of the coupling coefficients as published for ROS methods. It computes as `param.py:29-34`: `gammatilde = beta - alpha` or `gammatilde = gamma_ij`, `a = np.sum(alpha, axis=1)`, `g = np.sum(gammatilde, axis=1) + gamma`, `gammatilde = gammatilde / gamma`, then stores `alpha.T` and `gammatilde.T`. The legacy `rodas5p` table stores `gamma` on the diagonal of `beta` (`param.py:164-168`) and is used through `Rodas_param`, not through `from_hairer`. `interpolation` follows the resolution rule of Section 10.1, and `interp_order` defaults to 1 unless the class states it. Nothing else is written; `perform_step`, the controller, the norm and interpolation come from `Rosenbrock`, and the class is passed as `solve(dae, tspan, y0, alg=MyRos())`.

## 11. Events and callbacks

### 11.1 `ContinuousCallback`

```python
ContinuousCallback(condition, affect=None, *, direction=0, terminal=False, record=False,
                   rootfind='left', save_positions=(True, True), interp_points=10,
                   repeat_nudge=0.01)
```

- `condition(t, y, integ)` returns a float or a 1-D array of fixed length `m`.
- `direction` is an int or an array of length `m` with entries -1 (from positive to non-positive only), +1 (from negative to non-negative only) or 0 (both).
- `terminal` is a bool or a bool array of length `m`. `direction` and `terminal` are fixed for the callback; the adapter of Section 11.6 is a private subclass that takes them from each evaluation instead.
- `affect(integ, idx)` receives the int array of the callback's components whose root equals the event time. It may change `integ.u`, entries of `integ.dae.p` and `integ.dae.M.data`, and may call `integ.terminate()`.
- `record=True` logs each crossing as `(te, ye, ie)` into the result.
- `rootfind='left'` returns the last float before the crossing, but never the start of the step (Section 11.4); `'right'` returns the first float at which the component has crossed or is zero.
- A component is acting if it is terminal or the callback has an `affect`. A non-acting component of a recording callback is record-only.

### 11.2 `DiscreteCallback` and `preset_time_callback`

```python
DiscreteCallback(condition, affect, *, save_positions=(True, True), tstops=())
preset_time_callback(times, affect, *, save_positions=(True, True))
```

`condition(t, y, integ) -> bool` is evaluated after every accepted step, after the continuous callbacks. `tstops` are merged into the Integrator's heap, so the step ends exactly there. `preset_time_callback` builds `timeset = frozenset(float(x) for x in times)` once and returns `DiscreteCallback(lambda t, y, integ: t in timeset, affect, save_positions=save_positions, tstops=times)`; the membership test is exact because the step lands exactly on each `tstop`.

### 11.3 Detection on an accepted step

Each continuous callback keeps its bottom values `g0`, the condition at `(tprev, uprev)`: evaluated at initialization, and afterwards taken from the end of the previous accepted step or recomputed by the modification protocol. For an accepted step `[tprev, t]`:

1. `g1 = condition(t, u)`.
2. Component `i` is eligible if `g0[i] != 0` and its direction allows the sign of `g0[i]`: -1 needs `g0[i] > 0`, +1 needs `g0[i] < 0`. An exact zero at the start is therefore never an event, so nothing is reported at the initial point of a call, and a run that starts exactly on a root, as `Solverz/solvers/test/test_rodas_event.py:37-38` does, begins cleanly. The same rule means that a component that is exactly zero at the start of a step and moves away from zero in that step is not an event; its next crossing of zero is. `EventLoop` covers this case with its own row test (`SolPSDyn@c16a041:SolPSDyn/dae/event_loop.py:484-487`), and the user guide states it.
3. An eligible component crosses at the end if `g0[i] * g1[i] <= 0`, which includes an exact zero at `t`, unlike legacy's strict test (`rodas.py:232`).
4. Brackets. If some component is eligible and `interp_points >= 3`, the condition is evaluated on the interpolant at `t_k = tprev + k * (t - tprev) / (interp_points - 1)` for `k = 1 .. interp_points - 2`, with `t_0 = tprev` and the last point `t` itself (SciML semantics, `callbacks.jl:537-561`; with the default 10 there are 8 interior samples). For every eligible component, whether or not it crosses at the end, the bracket is `[t_{k-1}, t_k]` for the first `k` at which `g_k[i] * g0[i] <= 0`; a component with no such `k` has no crossing in this step. This finds a crossing that enters and leaves within one step, and the first of several crossings of one component, which the earliest crossing of D5 needs. With `interp_points < 3` the bracket of a component that crosses at the end is `[tprev, t]`.
5. Repeat nudge, only for `rootfind='left'`. For a component that fired at the start of this step, that is, whose last event time recorded by the modification protocol equals `tprev`, and whose bracket starts at `tprev`: the condition is evaluated at `tn = tprev + repeat_nudge * (t - tprev)`. If the component has crossed by `tn`, meaning `g(tn)[i] * g0[i] <= 0`, that crossing is the event already reported at `tprev` (F2), and the component has no event in this step. Otherwise its bracket bottom moves to `tn` (`callbacks.jl:523-532`).
6. No other crossing is discarded because it lies close to the start of the step; `opt.event_duration` is not read (D5).
7. Root finding and the skip rule. The acting components are root-found first, in increasing order of their bracket bottoms, ties by callback order and then by index. Let `te` be the smallest acting root found so far. Before a further component is root-found: if its bracket bottom is at or after `te`, it is skipped, since its root lies after `te`; if `te` lies inside its bracket, the condition is evaluated once at `te`, one vector call shared by every component for this value of `te`, and a component whose value at `te` has the sign of its bracket bottom is skipped, since it has not crossed by `te`; otherwise it is root-found in `[bottom, te]`. Record-only components are then treated in the same way against the final `te`, or all root-found when no acting component crosses. Every component whose root is at or before `te` is thus found, and D5's "every component crossing at `te`" holds, while components that cross later cost one evaluation instead of a root search.

Recorded at I6a. Step 2 alone loses a crossing that lies in the same step as a departure from an exact zero, since the component is then eligible nowhere in that step; this contradicts D5 and the statement of step 2 that its next crossing is reported, and a zero at a stop time followed by a crossing inside the next, longer step showed it. A component that is exactly zero at the start of the step therefore takes as its reference the first sample at which it is nonzero, if its direction allows that sign, and its bracket search starts from that sample; with `interp_points < 3` it has no bracket in that step, and a callback with such a component samples the step even when no component is eligible. In step 7 a `'left'` callback is tested at the float after `te`, and a `'left'` bracket is skipped only when it starts after `te`: a `'left'` root is the last float before the crossing, so a component whose crossing lies between `te` and the next float has its root at `te` while its value at `te` still has the sign of its bracket bottom, and the test at `te` itself would skip the twin of the component that set `te`, against the claim at the end of Section 11.4. Only acting or recorded components are bracketed, since a component that does neither has no effect; the condition's values are kept per time within a step, so each time costs one call per callback, shared by the components, and identical components cost no further call. The nudge of step 5 reads the components and the time that the modification protocol stores, so it takes effect with I6b. Step 5 cannot move the bottom to `tn` when `tn` lies at or after the top of the bracket, which needs `repeat_nudge >= 1/(interp_points - 1)`. If the component has crossed by `tn` it has no event in this step, as step 5 states; otherwise it has crossed and returned inside the nudge interval, that crossing is the event already reported, and its bracket is the first subinterval after `tn`, from `tn` and then between the sample points, at whose top it has crossed or is zero.

### 11.4 Root finding on the interpolant

`find_root(g, tl, tr, gl, gr, side, tstart)` for the scalar function `g(tau) = condition(tau, interp(tau))[i]`, with `gl != 0` and `gr == 0` or of the opposite sign, and `tstart` the start of the step:

```python
if gr == 0.0:
    return tr
moved = 0
for it in range(500):
    if np.nextafter(tl, np.inf) >= tr:
        break
    tm = tr - gr * (tr - tl) / (gr - gl)                     # regula falsi
    if it % 3 == 2 or not (tl < tm < tr):
        tm = tl + 0.5 * (tr - tl)                            # bisection every third step, and as fallback
        if not (tl < tm < tr):
            tm = np.nextafter(tl, np.inf)
    gm = g(tm)
    if gm == 0.0:
        return tm
    if np.sign(gm) == np.sign(gl):
        tl, gl = tm, gm
        if moved == 1:
            gr *= 0.5                                        # Illinois
        moved = 1
    else:
        tr, gr = tm, gm
        if moved == -1:
            gl *= 0.5
        moved = -1
if side == 'left' and tl > tstart:
    return tl
return tr
```

The bracket shrinks to adjacent floats with no tolerance parameter, as SciML's `abstol = reltol = 0` (`callbacks.jl:593-603`). `'left'` returns the last float before the crossing, the SciML default, suited to an `affect` that must act before the crossing. When that float is the start of the step, `'left'` returns the next float instead, the first at which the component has crossed or is zero. Every event time therefore lies strictly after the start of its step, so a call restarted from `(te, ye)` of a `'left'` event advances by at least one float and never reports an event at its initial point. `'right'` returns the first float at which the component has crossed or is zero. The bisection every third iteration bounds the count near that of plain bisection when regula falsi stalls. Components with the same function find the same pair of adjacent floats whatever their brackets, since the boundary between the two floats is a property of the function, so the skip rule of Section 11.3 does not separate identical components.

Recorded at I7b. The last sentence does not hold for a rounded interpolant. On the bouncing ball over `[0, 30]` at `rtol = 1e-6`, `Rodas4` crosses the ground in one step of 8.4, and its interpolant of the height changes sign at three pairs of adjacent floats within three floats of the impact, so two identical terminal components ended on different pairs and one of them was dropped. `_root` therefore gives `te` without a search to a component that has crossed by the probe time but not at the float before it, and `locate` locates again against `te` every acting component whose root lies after it, which can lower `te` further. `test_events.py` asserts the twins on that step, two terminal ones and a terminal one with a recorded twin, with each `rootfind`.

### 11.5 Handling, in `handle_callbacks`

1. Continuous callbacks. Let `te` be the smallest root of an acting component over all continuous callbacks, if any.
   - Without one: log every record-only crossing in increasing time, ties in increasing index, each with `ye = interp(root)`; the step stands; `g0 = g1` for every callback.
   - With one, `te` is one event instant for all callbacks:
     a. `u_e = interp(te)`; detach `u_step` (Section 5.4); set `t = te` and `u = u_e`, keeping `tprev`, `uprev`, `t_step`, `u_step` and `dt_step`, so that every later interpolation of this step still uses the step as computed.
     b. Log every record-only crossing with root `< te`, of any callback, in increasing time, ties in increasing index, each with `ye = interp(root)`; discard those after `te`.
     c. `policy.savevalues(integ)`, which saves the nodes `<= te` and, when every step is saved, the row `(te, u_e)`.
     d. Left save, once, if some callback with a component whose root equals `te` has `save_positions[0]` and the last saved time is not `te`.
     e. For each continuous callback, in list order, that has components whose root equals `te`: `idx` is those components, acting or record-only; if the callback records, log all of them in increasing index with `ye = u_e`, the state before any affect; call `terminate()` if any of them is terminal; call `affect(integ, idx)` if the callback has one. A callback with an `affect` has only acting components, so its `idx` holds no record-only component.
     f. If any `affect` ran, run the modification protocol of Section 11.7 once.
     g. Right save, once, if some processed callback has `save_positions[1]`.
     h. `dtpropose = dt_step`, the full length of the crossing step, SciML's `dtrelax = 1` (`callbacks.jl:668-748`).
2. Discrete callbacks, unless terminated: for each whose condition holds at `(t, u)`, in list order: `savevalues`, which saves the nodes `<= t` and, when every step is saved, `(t, u)`; left save if `save_positions[0]` and nothing was saved exactly at `t`; detach `u_step`; `affect(integ)`; the modification protocol; right save if `save_positions[1]` (`callbacks.jl:759-796`). When every step is saved, the step's own row at `t` is saved before the affect whatever `save_positions[0]` says, as in SciML, so `save_positions=(False, True)` gives two rows at `t` in that mode and one row with an output grid that does not contain `t`.
3. If no callback saved, `policy.savevalues(integ)`.

Recorded at I6b. Before the first `affect` of an event instant, and before the `affect` of a discrete callback, the core also runs the algorithm's `addsteps` if it has not run for the step, since an interpolation inside the step after the affect, which Section 10.5 allows, must describe the step as computed, and `addsteps` reads `p`, `M` and the end state, which the affect may change; this costs at most one Rodas3 end residual per event. In step 2, "unless terminated" is checked before each discrete callback, so an affect that terminates the run ends the list, and "nothing was saved exactly at `t`" means that the last saved row is not at `t`, the rule of step 1d, so that a second callback at the same `t`, or a discrete callback after a continuous event at `t`, adds no duplicate row before its affect. For the same reason the `savevalues` of step 2 runs once per step end, before the first discrete callback that fires, and not at all when the continuous callbacks have saved, since the legacy `savevalues` of the legacy-compatible configuration pushes `(t, u)` without looking at the last saved row; the rows of the default configuration are unchanged by this.

Simultaneous components are those whose separately located roots are the same float. Two components with the same condition, as in `test_rodas_event.py:67-72`, always are. Physically simultaneous crossings of different functions can come out one float apart; the later one is then reported by the next step or call.

### 11.6 The adapter for `opt.event`

`_LegacyEventCallback(event)` wraps `event(t, y) -> (value, isterminal, direction)` (`rodas.py:99-103`) as a private subclass of `ContinuousCallback` with `affect=None`, `record=True`, `rootfind='right'`, `save_positions=(False, False)`, and `interp_points=10`, the value of design rule 9. `condition` returns `value`; the `isterminal` and `direction` arrays are taken from the evaluation at the step end `(t, u)` of the step being examined. Consequences:

- A terminal component stops the run at the earliest terminal crossing time `te`, with `T[-1] == te` exactly and `Y[-1]` byte-equal to the logged `ye` row, in both configurations. Legacy stops at the first terminal component in index order instead (`rodas.py:233-300`).
- Every component whose root equals `te` is reported with `te` repeated, as `EventLoop`'s decoder needs: it matches `|te - T[-1]| <= 1e-12` and requires `te.size == ie.size` (`event_loop.py:238-288`).
- Non-terminal components are recorded without shortening the step. Legacy shortens the step at every recorded event and restarts from the interpolated state (`rodas.py:286-287`).
- With `'right'` the returned state has crossed or lies on the surface, so a new call from `(te, ye)` does not detect the same crossing again; this is what `EventLoop` does after a stop.
- With `len(tspan) > 2` a terminal stop returns `[t0, nodes <= te, te]` without a duplicate when `te` is a node; the final row is appended by `postamble` only if the last saved time differs from `te`.
- `te`, `ye`, `ie` are `None` when nothing was recorded, `ie` is int64, and `opt.event_duration` is ignored.
- Cost. On every accepted step with an eligible component, which for `EventLoop`'s vector of `3n` components per limiter site is nearly every step, the adapter calls `event` nine times, once at the end and at eight interior samples, and interpolates eight times, where legacy calls it once. Root finding adds one call per iteration for each component that crosses at or before `te`. I10 measures this against legacy (Section 17.1), and the user guide states it.

### 11.7 The modification protocol

It runs after an `affect`, once per event instant, and in `model_modified()`.

1. Detach `u_step` if not yet detached (Section 5.4). `integ.M = dae.M`, `integ.p = dae.p`, `model_epoch += 1`.
2. Reset the call's linear-solver cache as Section 8.4 states.
3. Inside `with linsolver(self._daeic_backend):`, `y = DaeIc(proxy, integ.u, integ.t, opts.rtol)`, copied into `integ.u`; any of the four `DaeIc` failures of Section 5.2 fails the run at `t`, with the rows saved so far, the left save included.
4. `new_step = True`, so `J0`, `F0` and `dF/dt` are recomputed; `_nl_eta = 1.0`; `alg.reset_history(integ, cache)`.
5. The bottom values of every continuous callback are recomputed at `(t, u)`; for `'left'` callbacks the components that fired and the event time are stored for the nudge.

Recorded at I6b. The components that step 5 stores for a `'left'` callback are those whose root equals `te` at the event instant, acting or record-only, since a record-only `'left'` root has not crossed either; a protocol run by a discrete callback or by `model_modified()` stores none and keeps what an event at the same `t` stored. Before the first step there is no accepted step to commit, so `model_modified()` copies the changed state into `uprev` itself, keeps the interpolant valid, which there covers only `t0`, and leaves the row at `t0` as the initial state; after a failure it does nothing.

The next loopheader commits the modified `u` into `uprev` (Section 5.4). This is the chain `u_modified -> reeval -> DaeIc` of design rule 8 (plan:93) and SciML's `reeval_internals_due_to_modification!` (`ODE@7393799:lib/OrdinaryDiffEqCore/src/integrators/integrator_interface.jl:66-91`), with two differences. SciML re-initializes the interpolated state before the affect, and the core does not, so a terminal adapter event returns the interpolated state as legacy does. SciML recomputes the interpolation data over the shortened step `[tprev, te]` (`change_t_via_interpolation!`); the core keeps the step as computed, through `u_step` and `t_step`, which gives the same interpolated values without a second step.

## 12. The author contract

### 12.1 `Algorithm`

```python
class Algorithm:
    scheme = None               # str, required: Stats.scheme and messages
    order = None                # int, required
    error_order = None          # default order; the error estimate is O(h**error_order), controller exponent 1/error_order
    interp_order = 1            # order in h of the interpolant's error on fixed-step runs, measured by the kit
    adaptive = False            # True if perform_step returns an error estimate
    explicit = False            # True if the method never solves with W
    inplace = False             # the style of perform_step
    norm = 'rms'                # 'rms' or 'max'
    legacy_compat = False

    def perform_step(self, s):                  # inplace = False: return y or (y, err)
    # def perform_step(self, integ, cache):     # inplace = True: write integ.u, set integ.EEst
    def alloc(self, integ):                     # default SimpleNamespace(); allocate buffers once
    def interpolant(self, integ, cache, theta, out):   # default linear between uprev and u_step
    def addsteps(self, integ, cache):           # default no-op; lazy data for interpolation
    def reset_history(self, integ, cache):      # default no-op; after the model or the state changed
    def controller(self, opts):                 # default IController(opts, self)
    def initial_dt(self, integ):                # default opts.dt0 or 1e-6*(tend - t0)
    def __call__(self, dae, tspan, y0, opt=None):     # legacy-shaped call
```

An algorithm is one class in one file, and its step is written once, in one style, as `perform_step`, the name issue #189 uses for both. `inplace = False`, the default, selects the formula style: `perform_step(self, s)` receives the `StepContext` of Section 12.2 and returns `y` or `(y, err)`. `inplace = True` selects the in-place style: `perform_step(self, integ, cache)` writes `integ.u` and, in an adaptive run, `integ.EEst`. `Integrator.__init__` raises `TypeError` when `type(alg).perform_step is Algorithm.perform_step`, or when the number of parameters of `perform_step` after `self` is not 1 for `inplace = False` and 2 for `inplace = True`; the result is cached per class. A subclass may change the style by setting `inplace` and overriding `perform_step`, and it inherits every hook, since the hooks have one signature in both styles: a formula variant of a table family is `class F(Rosenbrock): inplace = False` with its own `perform_step(self, s)`, and it reuses `alloc`, `addsteps`, `interpolant` and `controller`. Nothing is registered and nothing in the core changes. `Integrator.step()` advances the whole integration by one accepted step and is a different name from the algorithm's `perform_step`.

Hooks. `interpolant`, `addsteps` and `reset_history` take `(integ, cache)` in both styles and describe the last accepted step through `integ.tprev`, `integ.uprev`, `integ.t_step`, `integ.u_step` and `integ.dt_step`; they never read `integ.u`, which an event may have replaced (Section 5.4). `interpolant` may write into `out` and return `None`, or return an array, which the core copies into `out`, so formula code can write `return y0 + theta * (y1 - y0)`. `integ.ctx` gives formula code the services of Section 12.2 inside a hook. The default interpolant is `out = uprev + theta * (u_step - uprev)`, computed as `np.subtract(u_step, uprev, out=out)`, `np.multiply(out, theta, out=out)`, `np.add(uprev, out, out=out)`; it is safe for algebraic variables, and a better interpolant raises the accuracy of `saveat` and event location to its order.

Dispatch of the formula style: `res = alg.perform_step(integ.ctx)`; `y, err = res if isinstance(res, tuple) else (res, None)`; `TypeError` unless `y.shape == (n,)` and, when given, `err.shape == (n,)`, since `np.copyto` would broadcast a scalar silently; `np.copyto(integ.u, y)`. In an adaptive run, `err is None` raises `TypeError(f"{scheme}.perform_step returned no error estimate; set adaptive = False or run with opt.fix_h")`, and otherwise `integ.EEst = integ.error_norm(err)`. In a run that is not adaptive because `alg.adaptive` is false, an `err` that is not `None` raises `TypeError(f"{scheme}.perform_step returned an error estimate but declares adaptive = False")`; with `opt.fix_h` the estimate is ignored. Dispatch of the in-place style: `integ.EEst = None` before the call; after it, in an adaptive run that did not fail, `EEst is None` raises `TypeError(f"{scheme}.perform_step set no error estimate")`. These are programming errors.

An algorithm without an error estimate declares `adaptive = False`. It then runs with the fixed step `opt.hinit`, which is required, landing exactly on `tstops` and `tend`. A failed step in a fixed-step run fails the run, since the step cannot shrink.

`explicit = True` declares that the method uses `s.f`, the derivative `M^-1 F`, instead of solving with `W`. It is valid only for a model whose `M.nonzero()` pairs every row with exactly one variable and every variable with exactly one row, which is an ODE whose equations may be declared in any order, so `M` may be a scaled permutation. `Integrator.__init__` raises `TypeError(f"{scheme} is explicit and cannot integrate a model with algebraic equations or a singular mass matrix")` before `DaeIc` otherwise. Without this rule an explicit formula applied to `F` would move the variables by the residuals of the wrong rows, or move algebraic variables by the residuals of algebraic equations, without any message.

### 12.2 `StepContext` and the services

`integ.ctx` is a `StepContext` with `__slots__`, built once in `Integrator.__init__`. Its members are named after the symbols of the formulas.

| Member | Semantics |
|---|---|
| `s.t`, `s.h` | start of the attempt and its step, `integ.t` and `integ.dt` |
| `s.new_step` | true on the first attempt of a step, false on its retries |
| `s.y0` | a read-only view of `integ.uprev`, created once, since `uprev` is never rebound; valid during the attempt only, so an algorithm that keeps it copies it |
| `s.n`, `s.M`, `s.p` | size, mass matrix, parameters; never modify `M` |
| `s.D` | float64 vector with 1.0 on the rows of `M` that hold a nonzero value and 0.0 on the algebraic rows, those whose values are all zero, explicit zeros included; `s.D * v` keeps the differential rows of `v`; rebuilt when `model_epoch` changes |
| `s.rtol`, `s.atol`, `s.adaptive` | the tolerances of the run, and whether the run is adaptive |
| `s.cache` | the algorithm's cache from `alloc` |
| `s.F(t, y, out=None)` | residual; without `out` a fresh array is passed as `out`, so two results never alias, even for a module rendered before #187 |
| `s.F0` | a read-only view of `F(t, y0)`, evaluated at most once per step and counted once, shared with `s.dFdt` and the Rosenbrock stage 0 (Section 9) |
| `s.f(t, y, out=None)` | `M^-1 F(t, y)` through the one-to-one pairing of rows and variables, one counted residual; only for `explicit = True` |
| `s.J(t, y)` | Jacobian at any point, evaluated at every call |
| `s.J0` | the Jacobian at `(t, y0)`, evaluated at most once per step and kept across retries; the matrix of `s.W` |
| `s.dFdt(out=None)` | a read-only view of `dF/dt` at `(t, y0)` by the configuration's policy, computed once per step (Section 9) |
| `s.W(gamma)` | factorization of `M - (h*gamma) J0`, kept for the attempt; `.solve(b, out=None)` solves `W x = b` with row equilibration inside |
| `s.implicit(t, gamma, rhs, y=None, out=None, slope=False)` | returns `y` with `M y - h*gamma*F(t, y) = rhs` by the simplified Newton of Section 12.4, starting from `y` or `y0`; with `slope=True` it returns `(y, k)` with `k = (M y - rhs) / (h*gamma)`, which equals `F(t, y)` to the Newton tolerance and costs one matrix-vector product instead of a residual |
| `s.error_norm(e)` | the scalar of Section 10.4 for the vector `e` |

The same services exist on the Integrator as `integ.F`, `integ.F0()`, `integ.J`, `integ.J0()`, `integ.dFdt`, `integ.W`, `integ.implicit` and `integ.error_norm`, for in-place algorithms. Counting is automatic: `F`, `F0`, `f` and `J` count in `nfeval` and `nJeval`, factorizations and solves count in `ndecomp` and `nsolve`. An algorithm never touches `Stats`.

Out of place is the default. `out=` is accepted by `F`, `f`, `dFdt`, `W(...).solve` and `implicit` at any call site and is never required. With `out` none of them allocates, except for the backend result under SuperLU and the legacy-compatible dense path (Section 8.2).

Retries. On a retry of the same step, `s.t` and `s.y0` are unchanged, `s.new_step` is false, and `s.h` is the step the controller set after an error-test rejection, or half the previous step after a `StepFailure`. `s.F0`, `s.J0` and `s.dFdt()` return the values of the first attempt. `integ.u` holds the rejected trial state. `s.cache` holds whatever the failed or rejected attempt wrote, possibly only in part after a `StepFailure`.

History. When `s.new_step` is true, `s.cache` holds exactly what the last accepted attempt left in it, unless `reset_history` ran after it. The core keeps no data of earlier steps for an algorithm. An algorithm that uses data of the previous accepted step keeps it in `s.cache`, copies what it keeps, and clears all of it in `reset_history`, which the core calls in every modification protocol; the history check of the kit (Section 13) detects stale data. At the start of a call `alloc` gives an empty cache.

Recorded at I7a. "Only for `explicit = True`" is read as the use that `explicit` guarantees, not as a check on the algorithm: `s.f` works on any model whose `M` pairs rows and variables one to one and raises `TypeError` on any other, before it evaluates a residual, which is what the test of `s.f` in Section 15 asks for. `StepContext` binds the Integrator's services once, so the Integrator also has `integ.f` and `integ.D()`, which an in-place algorithm may use as well; with `slope=True`, either entry of the `out` pair of `implicit` may be `None`.

### 12.3 Styles

- Table: a Rosenbrock method is its tableau (Section 10.6). In C2 the Rosenbrock family is the only table-driven family.
- Formula: `perform_step(self, s)` written out of place with the services above; `ImplicitEuler` and `Trapezoid` below. An SDIRK method is a sequence of `s.implicit(..., slope=True)` calls, one per stage, sharing one factorization through `s.W(gamma)`. An explicit method declares `explicit = True` and uses `s.f`.
- In place: `inplace = True` and `perform_step(self, integ, cache)` with buffers from `alloc` and `out=` at every call site, as the built-in Rosenbrock family does.

### 12.4 The simplified Newton behind `implicit`

Solve `G(y) = M y - hγ F(t, y) - rhs = 0` with the Newton matrix `W = s.W(gamma) = M - hγ J0`. The iteration runs in buffers that the Integrator owns, `_nl_y`, `_nl_G`, `_nl_dz`, `_nl_w` and `_nl_w2`, so it allocates only the vector of `M @ y` per iteration and the backend result under SuperLU. In the code, `y_start` is the argument `y` of `implicit`:

```python
KAPPA, MAXIT = 0.01, 10
W = integ.W(gamma); hgamma = integ.dt * gamma
y = integ._nl_y; G = integ._nl_G; dz = integ._nl_dz; w = integ._nl_w; w2 = integ._nl_w2
np.copyto(y, integ.uprev if y_start is None else y_start)
eta = max(integ._nl_eta, np.spacing(1.0)) ** 0.8
ndz_prev = None
for k in range(1, MAXIT + 1):
    integ.F(t, y, out=G)                                    # one counted residual
    np.multiply(G, hgamma, out=G)
    np.subtract(M @ y, G, out=G)
    np.subtract(G, rhs, out=G)                              # G = (M y - hγ F(t, y)) - rhs
    W.solve(G, out=dz)                                      # one counted solve
    np.subtract(y, dz, out=y)
    np.abs(y, out=w); np.abs(integ.uprev, out=w2); np.maximum(w, w2, out=w)
    np.multiply(w, rtol, out=w); np.add(w, atol, out=w)     # atol + rtol*max(|y|, |y0|)
    np.divide(dz, w, out=w)
    ndz = np.sqrt(np.mean(np.square(w, out=w)))
    if not np.isfinite(ndz):
        raise StepFailure('the Newton iteration produced a non-finite value')
    if ndz == 0.0:
        break
    if ndz_prev is not None:
        theta = ndz / ndz_prev
        if theta >= 1.0:
            raise StepFailure('the Newton iteration diverged')
        eta = theta / (1.0 - theta)
        if ndz * theta ** (MAXIT - k) / (1.0 - theta) > KAPPA:
            raise StepFailure('the Newton iteration converges too slowly')
    if eta * ndz < KAPPA:
        break
    ndz_prev = ndz
else:
    raise StepFailure('the Newton iteration did not converge')
integ._nl_eta = eta
```

The result is copied out of `_nl_y`, into `out` when it is given and into a new array otherwise, since the next call of `implicit` overwrites the buffer. With `slope=True` the slope `k = (M y - rhs) / hγ` is written into a second new array, or into the second element of an `out` pair. The test is Hairer's, with `KAPPA` applied to the tolerance-weighted RMS norm as in SciML's `nlsolve`, so the Newton error is one percent of the tolerance. `J0` is evaluated once per step and kept across retries, and `W` is factorized once per `gamma` per attempt and kept across iterations and across `implicit` calls. A failure raises `StepFailure`, which the dispatch turns into `force_stepfail`, and the loop halves the step. Jacobian ageing across steps is left to milestone C3.

### 12.5 `ImplicitEuler`, as it ships

```python
"""Backward Euler written as its formula; the template of docs/src/integrator_adding_algorithms.md."""
from Solverz.integrator.algorithm import Algorithm

__all__ = ['ImplicitEuler']


class ImplicitEuler(Algorithm):
    r"""Backward Euler, ``M (y1 - y0) = h F(t0 + h, y1)``, of order 1.

    The algebraic equations hold at ``t0 + h`` by construction. The error estimate
    is the local error ``h**2 y''/2`` of the differential rows, written as
    ``(M (y1 - y0) - h D F(t0, y0)) / 2`` and passed through ``W = M - h J`` so that
    stiff and algebraic components are scaled. ``D`` removes the algebraic rows of
    ``F(t0, y0)``, where a residual that the consistent initialization left below
    its threshold would otherwise enter the estimate at every step size.
    """

    scheme = 'implicit_euler'
    order = 1
    error_order = 2
    adaptive = True

    def perform_step(self, s):
        y = s.implicit(s.t + s.h, 1.0, s.M @ s.y0)
        err = s.W(1.0).solve(0.5 * (s.M @ (y - s.y0) - s.h * (s.D * s.F0)))
        return y, err
```

### 12.6 `Trapezoid`, as it ships

```python
"""The trapezoidal rule written as its formula."""
from Solverz.integrator.algorithm import Algorithm

__all__ = ['Trapezoid']


class Trapezoid(Algorithm):
    r"""``M (y1 - y0) = h/2 (F(t0, y0) + F(t0 + h, y1))`` on the differential rows and
    ``0 = F(t0 + h, y1)`` on the algebraic rows, of order 2.

    ``f0 = D F(t0, y0)`` keeps the differential rows only. Averaging the algebraic
    equations as well would make their residual alternate in sign at a constant
    size, and an error estimate that carries it could never meet a tolerance below it.
    The error estimate ``M (y1 - y0) - h f0``, passed through ``W = M - h/2 J``,
    is the local error of the explicit Euler step, of order ``h**2``; it is
    conservative for the trapezoidal rule, whose own local error is of order ``h**3``.
    """

    scheme = 'trapezoid'
    order = 2
    error_order = 2
    adaptive = True

    def perform_step(self, s):
        f0 = s.D * s.F0
        y = s.implicit(s.t + s.h, 0.5, s.M @ s.y0 + 0.5 * s.h * f0)
        return y, s.W(0.5).solve(s.M @ (y - s.y0) - s.h * f0)
```

Both use the default linear interpolant with `interp_order = 1`, the default `IController` with exponent `1/2`, and the RMS norm. Both error estimates are filtered through the factorization that `implicit` has already computed, so each costs one solve, and `F(t0, y0)` is the step's shared `s.F0`.

On the algebraic rows the right-hand side of `implicit` is zero in both methods, because `M` has zero rows there and `D` removes the algebraic rows of `F0`. `implicit` therefore enforces `F(t0 + h, y1) = 0` on those rows, and both estimates are zero there, apart from the Newton error. `DaeIc` accepts an initial state whose algebraic residual has a 2-norm up to `1e-6` (`Solverz/solvers/daesolver/daeic.py:35-36`). Without `D`, that residual `delta` would enter the Trapezoid estimate as `2 J_aa^-1 delta` at every step and the ImplicitEuler estimate as `0.5 J_aa^-1 delta` at the first step, independently of `h`, and a run with `atol = 1e-10` would fail with too many rejections. The kit's inconsistent-start check (Section 13) guards this.

## 13. The conformance kit

`check_algorithm(alg, *, order_tol=0.3, rendered=False)` builds its models inline with `made_numerical(..., sparse=True)`, runs every check in order, raises `AssertionError(f"check_algorithm[{name}]: ...")` at the first failure, and returns a dict of the measured values. The same checks run on any algorithm; which models and step sizes they use depends only on the declared traits.

Models:

- A: variables `x`, `z`, `s`, `c` and a parameter `k = 1`, with `x' = -x + z`, `0 = z - k*s`, `s' = c`, `c' = -s`, from `x = z = s = 0`, `c = 1`. The forcing is carried by an oscillator so that the model is autonomous: `dF/dt` is then exactly zero and cannot limit the measured order. Exact: `x(t) = (exp(-t) + sin t - cos t)/2`.
- P: A with the equations declared in the order `c' = -s`, `0 = z - k*s`, `x' = -x + z`, `s' = c`, so that the rows of `M` are not aligned with the variables.
- E: A with `z` eliminated, `x' = -x + k*s`, `s' = c`, `c' = -s`, declared in the order `c'`, `x'`, `s'`, so that `M` is a permutation. When `alg.explicit`, every check below that names A or P runs on E instead, and the inconsistent-start check is skipped, since E has no algebraic equation.
- A_delta: A started from `z = 1e-7`, whose algebraic residual `1e-7` lies below `DaeIc`'s threshold and is left in place (`daeic.py:35-36`).
- B: the bouncing ball of `test_rodas_event.py:7-12`, `x0' = x1`, `x1' = -9.8`, from `[0, 20]`; the first impact is at `40/9.8`.
- C: `x' = x**2` from `x = 1` on `[0, 2]`, which blows up at `t = 1`.

When `alg.adaptive` is false, every run passes `hinit = h_fix = 2**-6`, and the checks whose criterion depends on error control say so below.

| Check | Run | Pass criterion |
|---|---|---|
| contract | construction | `perform_step` defined with the signature of `inplace`; `order` a positive int; `error_order` a positive number; `scheme` a str; if `explicit`, a `TypeError` on A |
| order | A and P, or E when `explicit`, with `opt = Opt(fix_h=True, hinit=h, rtol=1e-12, atol=1e-14)`, `h = 2**-k`, `k = 3..6` | on each model, the mean of the two finest slopes `log2(e_k / e_{k+1})` of `|x(1) - x_exact(1)|` is at least `alg.order - order_tol` |
| interpolant | the runs of the order check, through `init` and `step()`; in the last step, `integ.interp` at `theta` in 0, 0.25, 0.5, 0.75 and 1 | at `theta = 0` it equals `uprev` exactly; at `theta = 1` `max|y - u| <= 1e-10 * (1 + max|u|)`; the mean of the two finest slopes of the largest interior error against the exact solution is at least `alg.interp_order - order_tol` |
| saveat | A, `rtol=1e-6`, `atol=1e-8`: once with `[0, 1]` and once with `np.linspace(0, 1, 11)`, both through `init` and `step()`; in the first run, `integ.interp(node)` is recorded right after the step that contains each node | the lists of accepted `t` are byte-equal; `T` of the second run is byte-equal to the `linspace`; every row of the second run is byte-equal to the recorded `interp` value |
| tstops | A with `tstops=[0.25, 0.5, 0.75]` and `[0, 1]` | `T` contains the three values byte-exactly, `T` increases strictly, `T[-1] == 1.0` |
| events | B with `Opt(event=...)`, value `[x0, x0]`, terminal `[1, 1]`, direction `[-1, -1]`, `[0, 30]`, `rtol=1e-6`, `atol=1e-8` | `ie == [0, 1]`, `te[0] == te[1]`, `abs(te[0] - 40/9.8) <= 1e-3 * 40/9.8`, or `<= 10 * h_fix` without error control, `T[-1] == te[0]`, `Y[-1]` byte-equal to `ye[0]`, no event at `t0` |
| events on a grid | B as above on `np.linspace(0, 30, 61)`, and B without `opt.event` on the same grid | every row at a node `< te` is byte-equal between the two runs, since the step sequence up to the crossing step is the same and the crossing step is interpolated as computed |
| inconsistent start | A_delta with `rtol=1e-6`, `atol=1e-10` on `[0, 1]` | `ret == 'success'` |
| history | A with `fix_h=True`, `hinit=2**-6` on `[0, 1]` and a `preset_time_callback` at 0.5 that sets `k = 2`; against two calls on `[0, 0.5]` and `[0.5, 1]`, the second from the last row of the first with `k = 2` | every row after 0.5 byte-equal between the two, on either backend; a difference means the algorithm kept data across the modification |
| failure | C | no exception; `ret == 'failed'`, `succeed is False`, `T[-1] < 1`, `T.size >= 2`; exactly one line on stdout |
| out | A with `dae.F`, and with `nDAE(M, lambda t, y, p: F(t, y, p), J, p)` | byte-equal `T` and `Y`, equal `Stats` counters |
| Opt | A ten times with one `Opt` | every field of `vars(opt)` unchanged after every call; the ten `T` and `Y` byte-equal |
| y0 | A with an ndarray `y0` and with a `Vars` | both byte-equal after the call |
| counts | A with counting wrappers on `F` and `J` | `stats.nfeval` and `stats.nJeval` equal the observed calls; `stats.nstep == len(T) - 1`; `ret == 'success'` |
| rendered | only with `rendered=True`: A rendered with `module_printer(..., jit=True)` into a temporary directory, both on `np.linspace(0, 1, 11)` | the criterion of Section 15 for rendered against inline |

The models have fewer than 1000 unknowns, so the repeatability checks hold on both backends (Section 8.4), and the history check holds on both because the modification protocol resets the solver cache to the state of a new call.

Recorded at I7a, for I7b. On A the first-order error coefficient of `ImplicitEuler` at `t = 1` is 0.017, against 0.125 at `t = 0.5`, so for `k = 3..6` the error `|x(1) - x_exact(1)|` is ruled by its `h**2` term and gives the slopes 3.76, -1.38 and 0.50, and the order check as written fails for `ImplicitEuler`, on P identically. The largest error over the saved rows of the fixed-step run gives 0.875, 0.940 and 0.971 for `ImplicitEuler` and 2.00 for `Trapezoid`; `test_services.py` measures that, and I7b needs a criterion of this kind.

Recorded at I7b. The bracket of the message names the check, as in `check_algorithm[order]: rodas4: ...`, and an exception raised inside a check fails that check. A class instead of an instance raises `TypeError`, and an instance with `legacy_compat=True` raises `ValueError`, since that configuration refuses `tstops`. The contract check also requires `interp_order` to be a positive number, since the interpolant check reads it.

The order and interpolant checks take the largest error of `x` over the saved rows, and over the points 0.25, 0.5 and 0.75 of every step, of the runs with `h = 2**-k` for `k = 1..6`, and the mean of the two finest slopes whose finer error is at least `1e-12`. The errors of `Rodas5P` on A for `k = 3..6` are `3.2e-11`, `5.1e-13`, `1.2e-14` and `6.2e-15`, so the two finest slopes of the table lie at the rounding level. The two ends of every step are evaluated through the algorithm's interpolant, since `interp` returns `uprev` and `u` themselves there. Measured on A, the interpolant orders are 3.00 for `Rodas3`, 3.94 for `Rodas4`, 3.99 for `Rodasp`, 4.93 for `Rodas5P`, 0.97 for `ImplicitEuler` and 1.97 for `Trapezoid`, none below the declared value.

The saveat check adds a third run on `[0, 1]` that interpolates nowhere, whose accepted times must equal those of the other two. Both runs of the table interpolate at the same nodes, so only such a run shows an `addsteps` or an interpolant that changes the steps. The events-on-a-grid check adds to the grid a node halfway between the start of the crossing step and `te`, both from the run without a grid, since the 61 nodes need not fall inside the crossing step before `te`, and only such a node shows an interpolant that reads `u`, which the event has moved to `te`. The contract check accepts a `TypeError` on A for an explicit algorithm only with the core's message, so that another error of the algorithm does not pass as the refusal. The events check also requires `ret == 'terminated'`. The inconsistent-start check asserts that `DaeIc` has left `z = 1e-7` in place, without which it tests nothing. The y0 check also requires the two runs to give one trajectory.

Model C is `x' = x**3` from 1 on `[0, 1]`, which blows up at 0.5 with no real continuation. The error estimate of `Rodas3` vanishes on `x' = x**2`, at most `1.2e-11` at every step, so it steps over the pole in 9 steps and returns `x(2) = -1` with `ret == 'success'`. On `x' = x**2`, `Rodas4` at `rtol = 1e-6`, `Rodas5P` and explicit Euler with `h_fix` also fail only after `t = 1`, since the numerical blow-up lags the exact one. With `tend = 1` the criterion `T[-1] < 1` states that the run ends before `tend`.

Where the table gives no `Opt`, namely for tstops, failure, out, Opt, y0 and counts, the runs use `Opt()`, with `hinit = h_fix` without error control. The rendered check runs at `rtol=1e-6`, `atol=1e-8`, and `rendered_matches_inline` in `testing.py` is the criterion of Section 15, which `test_rendered_inline.py` imports.

## 14. Legacy Rodas deprecation

The only change to `Solverz/solvers/daesolver/rodas/rodas.py` is the warning: one helper function above `Rodas` and one statement inserted as the first statement of the body of `Rodas`, before `if opt is None:` at line 65, with the imports `sys` and `os` that the helper needs.

```python
_NOT_CALLER = (os.path.abspath(__file__), os.path.abspath(dae_io_parser.__code__.co_filename))

def _caller_stacklevel():
    """Stack level of the first frame outside this module and the dae_io_parser wrapper."""
    frame, level = sys._getframe(1), 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename) in _NOT_CALLER:
        frame, level = frame.f_back, level + 1
    return level

    # first statement of Rodas:
    warnings.warn("Rodas is deprecated. Use Rodas3, Rodas4, Rodasp or Rodas5P from "
                  "Solverz.integrator, called as Rodas4()(dae, tspan, y0, opt); the class "
                  "selects the method, not opt.scheme. Rodas4(legacy_compat=True) and the "
                  "other three reproduce Rodas on adaptive runs without events, given the "
                  "class that matches opt.scheme and an Opt that no earlier Rodas call has "
                  "changed; the user guide lists the conditions.",
                  DeprecationWarning, stacklevel=_caller_stacklevel())
```

`warnings` is already imported (`rodas.py:1`), and `dae_io_parser` reaches the module through `from Solverz.solvers.daesolver.utilities import *` (`rodas.py:3`). The helper counts from the frame of `Rodas`, level 1, and skips every frame whose file is `rodas.py` or the file of the `dae_io_parser` wrapper (`Solverz/solvers/parser.py:82-100`). Through the wrapper it returns 3, the caller's line. A caller that calls `Rodas.__wrapped__` directly, as some scripts in SolAlg's repository do (`Solverz-beta:SolAlg/daesolver/AdamsBDF/_test_ns.py:17-20`), gets 2, again the caller's line; a fixed `stacklevel=3` would point one frame above it. A fixed `stacklevel=2` would point at `parser.py`, where Python's default filter hides a `DeprecationWarning` outside `__main__`. `warnings.warn(..., skip_file_prefixes=...)` would do the same from Python 3.12, and Solverz supports Python 3.10 (`pyproject.toml:35`). The text names only the replacements and says nothing about removal, since legacy solvers are never deleted (D1). The docstring and every other line stay unchanged; no other legacy solver gets a warning.

Recorded at I8. The docstring of `Rodas` stays its first statement, since a statement before it would turn it into an expression and empty `Rodas.__doc__`, which the reference documentation renders; the warning is the first statement after the docstring, before `if opt is None:`.

Readable test output. The repository's own tests call legacy Rodas on purpose: `tests/test_dae.py:22-41` (at import time), `tests/test_rodas_dense.py:24-25`, `tests/test_rendered_F_aliasing.py:85-86, :169-170`, `Solverz/solvers/test/test_radau.py:42, :56, :70`, `Solverz/solvers/test/test_pe_adams_bdf.py:132`, `Solverz/solvers/test/test_rodas_event.py:29, :74-75`, and the parity tests of Section 15. `pyproject.toml:44-45` gains one line, so the table becomes:

```toml
[tool.pytest.ini_options]
addopts = "-vv"
filterwarnings = ["ignore:Rodas is deprecated:DeprecationWarning"]
```

The pattern matches the start of the message only. `pytest.warns` records warnings regardless of this filter, so `tests/integrator/test_deprecation.py` still sees the warning. The filter also applies during collection, where `tests/test_dae.py` integrates. No configuration in the repository turns warnings into errors (only per-test `ignore:Equation size` marks exist, `tests/test_indexed_symbol_identity.py:65`, `tests/test_unsound_loop_jacobian.py:18`), and none in the local SolMuseum and Cookbook checkouts, so downstream suites show the warning in their summaries without failing.

## 15. Test plan

All test files live in `tests/integrator/` with an `__init__.py`. Data files are located relative to `__file__`. Rendered models are built by module-scoped fixtures with `module_printer(..., jit=True)` into `tmp_path_factory` directories with unique module names, following `tests/test_rendered_F_aliasing.py:45-54`. Tests whose subject is the backend are parametrized over `'klu'` and `'superlu'` through the `linsolver` context manager, and the `'klu'` case is skipped when `KLU_AVAILABLE` is false; every other test uses the global backend, so the Windows CI runs all of them on SuperLU (`.github/workflows/ci-cd.yml:30-37`). The fixture `klu_matching_low` calls `set_klu_matching(True, min_n=2)` and restores the previous `klu_backend._MATCHING` and `klu_backend.MATCHING_MIN_N` afterwards, so that small models exercise the value-dependent row matching.

Model variants. A model is built as inline sparse, inline dense or rendered sparse; `module_printer` has no dense option, so no rendered dense variant exists.

Shared helpers:

- `tests/integrator/models.py`: builders returning `(dae, y0)` for each variant. Models: `dae_test` (`x' = -x**3 + 0.5 y**2`, `0 = x**2 + y**2 - 2`, `tests/test_dae.py:6-13`); `forced`, a hand-built `nDAE` with `x' = -x + z`, `0 = z - (1 + sin t)`, from `x = 0`, `z = 1`, `M` a `csc_array`, `F` written with `np.sin(t)` and `J` returned as a `csc_array` or as an ndarray, inline only, because Solverz has no symbol for time and explicit time reaches a generated residual only through `TimeSeriesParam` (`Solverz/code_printer/python/utilities.py:97-100`); `trace` (`x' = -x + u(t)`, `0 = z - x u(t)` with `TimeSeriesParam('u', v_series=[0, 1, 1, 0.5], time_series=[0, 0.1, 0.5, 1.0])`), which carries the non-autonomous coverage of the rendered variant; `vdp` (Van der Pol with `mu = 10` from `x = [2, 0]`, `Solverz/solvers/test/test_radau.py:18-24`, whose many rejected attempts exercise the retry path); `alloc(n)` (`x' = -k x + sin x`, `x = linspace(0.5, 1.5, n)`, `k = linspace(0.9, 1.1, n)`, as in a private benchmark record); `ladder(n)` (`x' = Mat_Mul(A, x) + 0.1 sin x` with `A` the sparse tridiagonal matrix of `-2` on the diagonal and `1` beside it, whose `W` is not diagonal); `ball` and `orbit` (`test_rodas_event.py:6-72`); `permuted` (the algebraic equation first and the Ode second, so rows and variables of `M` are not aligned). Recorded at I2: `build(name, variant, *args, directory=None)` returns `(dae, y0)` with `y0` a new float64 array, and the builders ending in `_model` return the symbolic model and its `Vars`; `trace` starts from `x = 0.5`, `z = 0`, `ladder(n)` from `x = linspace(0.5, 1.5, n)`, and `permuted` is `dae_test` with its two equations swapped. The session fixture `model` of `tests/integrator/conftest.py` builds each model once.
- `tests/integrator/legacy_rodas.py`: the transcription of D3a. `legacy_step(dae, M, p, t, y0, dt, J, rparam, opt, linsolver, klu_cache, state) -> (ynew, err_raw, K)` copies `rodas.py:156-212` line by line with `J` passed in, where `err_raw` is `err` after the `1e6` override and before the floor, and `state` carries `rscale_to_dense`. `legacy_run(dae, tspan, y0, opt, attempts=None) -> (T, Y)` copies `rodas.py:65-358` without the event branch and calls `legacy_step`; with a list `attempts` it appends `(t, y0.copy(), dt, reject, ynew, err_raw)` for every attempt. The recorded `dt` is the step after the stretch of `rodas.py:146-149`, taken immediately before `K = np.zeros(...)`, since that is the step the attempt uses. `t` and `dt` are stored unconverted: `t` is `np.int64` on the first attempt of an integer `tspan` such as `[0, 20]`, and `dt` can be `np.int64` when the stretch fires on that attempt. It imports `dfdt` and `ntrp` from the legacy module, since those are the legacy code. Recorded at I2: each record is the named tuple `Attempt` with `J`, the Jacobian the attempt used, as a seventh field, since the lockstep of I3 takes `J` from the record. `legacy_step` raises `LegacyStop` where legacy leaves its loop silently on a factorization error, and `legacy_run` omits the progress bar and the counters, which change no state, and refuses `opt.event`.

Each test file below lists the milestone whose gate it belongs to; a file whose tests need two milestones marks each test, and the gate of a milestone is the set of tests marked with it. Recorded at I2: the markers are `i2` to `i8`, registered in `tests/integrator/conftest.py`, so that `-m i2` selects the tests of the I2 gate. A file of one milestone from I2 on carries its marker at module level through `pytestmark`, and a file shared by two milestones carries it on each test; the files of I1 predate the markers and are selected by name.

| File | Milestone | Asserts |
|---|---|---|
| `test_import.py` | I1 | importing `Solverz.integrator` imports no legacy module in a cycle: each module of the package imports at module level only the Solverz modules of Section 3.4, no module outside the package except `Solverz/__init__.py` imports it, and each imports first in a fresh interpreter; `from Solverz.integrator import *` binds exactly `__all__`; after `from Solverz import *` no name `solve` or `init` exists, and after `from sympy import *` it leaves SymPy's `solve` bound |
| `test_options.py` | I1 | added at I1: the mapping of Section 4.2, `Opt` unchanged, the types of both configurations, read-only copies of the nodes and of an array `atol`, and every `ValueError` of `from_opt` with its text |
| `test_saving.py` | I1 | added at I1: `to_daesol` gives new float64 `T` and C-contiguous `Y`, `None` events or float64 `te` and `ye` and int64 `ie`, and the `Vars` conversion of Section 4.4 |
| `test_algorithm_defaults.py` | I1 | added at I1: the traits and default hooks of `Algorithm`, the style check of Section 12.1, `StepFailure` distinct from the exception types the core catches otherwise, and the members of the `StepContext` stub |
| `test_matmul_out.py` | I1 | `np.matmul(K, v, out=buf)` byte-equal to `K @ v` for C-contiguous `K` of shape `(n, s)`, `n` in 2, 28, 1000, 10000, `s` in 4, 6, 8, with `v` a transposed-table column view and a contiguous vector, and with `buf` prefilled with `NaN` and with `+inf` |
| `test_legacy_transcription.py` | I2 | `legacy_run` output byte-equal to legacy `Rodas` for the four schemes, `[t0, tend]` and a grid of 201 nodes, on `dae_test` (`[0, 20]`, `hinit=0.1`) in all three variants, `forced` (`[0, 1]`, `rtol=1e-6`, `atol=1e-8`) with sparse and dense `J`, `trace` (same tolerances) in all three variants, and `vdp` (`[0, 20]`, `rtol=1e-6`, `atol=1e-9`, inline sparse); each backend. Under `klu_matching_low`, `permuted`, whose row matching is not the identity, and `ladder(40)`, whose matching is the identity but which exercises the analysis with the matching and the permuted solve on a tridiagonal pattern; and rendered `alloc(1000)` with Rodas4 under KLU. The KLU cases are skipped when KLU is not available. Grids stay at or below 10000 nodes, since a larger grid can overflow the legacy buffer (`rodas.py:340-344`). Recorded at I2: `permuted` runs as `dae_test`, `ladder(40)` on `[0, 2]` at `rtol=1e-6`, `atol=1e-8`, and rendered `alloc(1000)` on `[0, 1]` with `hmax=1e-2`, each also on 201 nodes. The iteration matrix of `ladder(40)` is diagonally dominant, so its matching is the identity at every attempt; the test asserts the non-identity permutation of `permuted`. The records are checked field by field against the same run, since I3 consumes every field: `t` chains through `t + dt` of the accepted records and ends each at its saved time, `J` equals the Jacobian at `(t, y0)` on the first attempt of a step and is the same object on a retry, and `trace` records an accepted `err_raw` below `1e-6`, which shows that the error is taken before the floor |
| `test_linalg.py` | I2, I5 | I2: the built `Wm` equals the legacy chain byte for byte in `data`, `indices`, `indptr` (sparse) and array (dense); `J`'s `data`, `indices`, `indptr` unchanged after a build; legacy-compatible calls start from an empty `KLUCache`; `solve(b, out=...)` byte-equal to `solve(b)` on each backend, with and without a matching; a strided `out` raises `ValueError`; a singular `W` and a failing `klu_solve` raise `StepFailure`; `ndecomp` and `nsolve` counts. I5: a default second call on the same model below the threshold reuses `model_cache(dae).symbolic`, the same object when the pattern is unchanged; under `klu_matching_low` a second call analyses anew, and two calls from the same state are byte-equal; `cache.superlu` starts empty in every call; the default dense factorization agrees with `np.linalg.solve` to `1e-12` relative. Recorded at I2: the models' Jacobians are canonical, so the check that `J` is unchanged also runs on a hand-built `J` with unsorted rows and a stored exact zero whose `indices` and `indptr` another `J` shares, on which `sort_indices`, `eliminate_zeros` and `sum_duplicates` each change bytes, after the build and after the factorization |
| `test_derivative.py` | I2, I4 | I2: `'legacy'` byte-equal to legacy `dfdt(dae, t, y)` at `t` in 0, 0.3, 17 on `forced` and `trace`; `'ode23s'` follows its formula exactly for independently computed `tdel`; on `forced` at `t` in 0 and 0.3 and `dt` in `1e-3` and `5e-8`, with `F_t = [0, -cos t]` exact, `|dt * (ft - F_t)| <= 4 * SQRT_EPS * max(1, max|y|)` on every row for `'ode23s'`; at `t = 0`, `dt = 1e-3` the `'legacy'` value violates that bound on the algebraic row, so the test tells the two policies apart. I4: `F0` and `dF/dt` are evaluated once per step and reused on retries, by counted residuals on `vdp` |
| `test_rosenbrock_lockstep.py` | I3 | D3a: for every attempt recorded by `legacy_run` on the matrix of `test_legacy_transcription.py`, the kernel in the legacy-compatible configuration, with `integ.t` and `integ.dt` assigned the recorded values unconverted, `y0` and `J` from the record, `new_step = (reject == 0)` and its own `IterationMatrix` fed the same sequence, gives `integ.u` byte-equal to `ynew` and `integ.EEst` byte-equal to `err_raw`, rejected attempts included. Recorded at I3: the attempt runs through the Integrator's dispatch, the recorded Jacobian is returned by the model's `J`, which asserts that it is asked for at `(t, y0)` of the record; each attempt also checks its counters, `s + 1` residuals, one Jacobian, one factorization and `s` solves on the first attempt of a step and `s - 1` residuals on a retry; on a grid, each accepted attempt is saved as the legacy-compatible configuration saves it, through `addsteps` once and `interpolant` at every node it covers, and every row is byte-equal to legacy's |
| `test_rosenbrock.py` | I3 | added at I3: the built-in methods are the tables of `Rodas_param`; `from_scheme` with its errors; `from_hairer` gives the `rodas4` arrays bit for bit with `beta` and with `gamma_ij`, refuses malformed tables, and a table-only subclass takes legacy's attempts; the traits a subclass takes from its tableau; the legacy norm with its `1e6` override and its `NaN` from `0/0`, and the two default norms, each byte-equal to its formula; the default configuration takes legacy's step on the autonomous `vdp`, where both `dF/dt` policies give zero; the Rodas3 slopes in both configurations, the pairing through `M` with explicit zeros and repeated rows or columns, and its rebuild when `model_epoch` changes; the contract of `interp`; `F0`, `J0` and `dFdt` kept on a retry and after acceptance; the dispatch of both styles, with a `StepFailure` from `F` or from a singular `W`, a `FloatingPointError` under `np.errstate(all='raise')`, and every `TypeError` of Section 12.1; the residual service, its adapter, and its conversion of arithmetic errors into `StepFailure` while other exceptions propagate |
| `test_compat_parity.py` | I4 | D3b: `RodasX(legacy_compat=True)(dae, tspan, y0, Opt(...))` byte-equal in `T` and `Y` to legacy `Rodas` with its own identical `Opt`, over the whole matrix of `test_legacy_transcription.py` and both backends; negative control: two consecutive calls on the same model from different initial states, each byte-equal to its own legacy call |
| `test_controllers.py` | I4 | hand-computed `q`, `dtnew` and state transitions of `IController`, `PIController`, `LegacyRodasController`, including `EEst = 0`, `inf`, `NaN`, the cap after a rejection that lasts until the next acceptance, and bounds |
| `test_failure_and_opt.py` | I4, I5 | I4, in the legacy-compatible configuration and with in-place test algorithms: blow-up model: prefix, `ret == 'failed'`, `succeed is False`, `stats.t_fail`, no raise, one printed line; more than 100 rejections through a test algorithm whose error is always `1.0001`; a test algorithm that always raises `StepFailure`; an algebraic equation without a real solution (`0 = z**2 + 1`), whose `DaeIc` failure at `t0` gives `T == [t0]`, `ret == 'failed'` and no raise, sparse and dense; a residual that raises a `ValueError` of its own at `t0` propagates; a residual that raises `ZeroDivisionError` from `t = 0.5` on fails with `ret == 'failed'` and no raise; ten calls with one `Opt`; `y0` unchanged (ndarray and `Vars`); `ValueError` for `fix_h` without `hinit`, for `t0 > tend`, for `hinit <= 0` and for `tstops` in the legacy-compatible configuration; `t0 == tend` gives one row and `'success'`; counters equal counting wrappers, including `DaeIc` and `dF/dt` calls. I5: an `nDAE` whose `F` calls an `@njit` function that computes the scalar `1.0 / (t - 0.5)`, with `tstops=[0.5]`, so that a residual at `t = 0.5` raises Numba's `ZeroDivisionError`, fails with `ret == 'failed'` and no raise, and so does the same `F` without `@njit`, where NumPy returns `inf` for the `np.float64` time; `opt=None`, `alg=None`; the `opt.scheme` warning appears once with `scheme='rodas3'` and `Rodas4()`, for each of the three entries, at the caller's line, and not with `Opt()`. Recorded at I5: the default configuration passes a stop time to `F` as a Python float, whose division by zero raises `ZeroDivisionError` in plain Python as well, so the test's `F` converts the time with `np.float64` before it calls the function, and the variant without `@njit` then returns `inf` as stated |
| `test_dae_xlsx.py` | I4, I5 | on `tests/dae_test.xlsx`: I4: `Rodas4(legacy_compat=True)` within `1e-8` of the sheets `rodas` and `rodas_dense` as `tests/test_dae.py:44-48`. I5: `Rodas4()` in the default configuration on `np.linspace(0, 20, 201)` within `1e-2` of `rodas_dense`, with the maximum deviation printed; the `rodas` sheet holds the 9 rows of legacy's own accepted steps without a time column (`tests/test_dae.py:22-25`), so the default configuration, which takes other steps, is not compared with it |
| `test_saveat_tstops.py` | I5 | nodes byte-exact; accepted `t` sequence identical with and without a grid; `tstops` byte-exact; `T[-1] == tend` exactly for spans such as `[0, 0.3]` and `[0.1, 0.7]`; `tstops` outside `(t0, tend)` ignored; after a truncated step the next `dt` is at least the untruncated one; a step within `100` units in the last place below a `tstop` is stretched to it; `interp` accepts `tq` in `(tprev + dt_step, t]` after such a step; Rodas3 on `permuted`: node values within `1e-5` of a `Rodas5P` run at `rtol=1e-11`, `atol=1e-13`, and the algebraic variable within `1e-12` of the linear interpolant between `uprev` and `u_step`. Recorded at I5: the Rodas3 run uses `rtol=1e-8`, `atol=1e-10`, since the linear interpolant of the algebraic variable has an error of order `h**2`; at `rtol=1e-6` the nodes of the first steps deviate by `4.5e-05` while the step ends deviate by `2.4e-07`, and at `rtol=1e-8` the nodes deviate by `2.3e-06`, against `1.1e-04` for the legacy-compatible interpolant without the pairing, which the test asserts as a control |
| `test_rendered_inline.py` | I5 | per model and scheme, rendered and inline runs are each byte-equal to their legacy counterpart in the legacy-compatible configuration (covered by `test_compat_parity.py`, referenced); between rendered and inline in the default configuration on `dae_test` over `np.linspace(0, 20, 201)` at `rtol=1e-6`, `atol=1e-8`: accepted step counts differ by at most 2 and `max|dY| <= 10 * rtol`, with both numbers printed. This is the one criterion for rendered against inline, used by the kit too. Byte equality between rendered and inline is not required, because Numba and NumPy evaluate `**` and transcendental functions with different routines (Section 2.3) |
| `test_events.py` | I6a, I6b | I6a: ten calls on `ball` with `Rodas4()` and `Opt(event=...)` on `linspace(tstart, 30, 100)` as `test_rodas_event.py:20-46`: `te` within `rtol=1e-5` of the ten legacy values; `orbit` with each of the four schemes at `rtol=1e-8`, `atol=1e-10`: `te` and `ye` within `1e-7` relative of a `Rodas5P` run at `rtol=1e-11`, `atol=1e-13`, and `ie == [1, 0]`; the legacy values of `test_rodas_event.py:77-97` were computed at `rtol=1e-5`, `atol=1e-4` and encode that run's error, so they are not asserted. Two identical components give `te` repeated and `ie == [0, 1]`; `T[-1] == te` and `Y[-1]` byte-equal to `ye[-1]`, in both configurations, including a terminal crossing at a step end that is also a node in the legacy-compatible configuration; with a grid, the output is `[t0, nodes <= te, te]` with no duplicate when `te` is a node (`condition = t - 0.5`, node 0.5); a terminal event stops the legacy-compatible configuration; no event at a start on a root; a component exactly zero at the start that then turns negative is not reported, and its next crossing is; a crossing `1e-10` after the start of a long first step is found; a terminal condition with three crossings inside one step (`hmax = 1`, a cubic in `t` on a linear model) stops at the first; an exact zero at a step end is reported once; the direction filter; `te`, `ye`, `ie` are `None` without events and `ie.dtype == np.int64`; a crossing that enters and leaves within one step (`g = (t - 0.5)**2 - 1e-4` with `hmax = 1` on a linear model) is found; a terminal `'left'` callback restarted from `(T[-1], Y[-1])` with the state unchanged gives `te > t0` and `T.size >= 2`. Recorded at I6a: at `rtol=1e-8` the orbit runs of Rodas3 and Rodas5P without events already deviate from the reference by `9.3e-7` and `4.3e-7` of the state at the event times, so `1e-7` cannot hold whatever the event location; the test asserts `1e-6` for `te` and `1e-5` for `ye`, measured relative to the largest entry of its row since `y[1]` at the return is about `1e-9`, against measured maxima of `1.1e-7` and `1.8e-6`, and a further test asserts that `te` is the first float at which the terminal component has crossed on the interpolant. The crossing that enters and leaves runs with `hinit = 0.9`, so that the first step `[0, 0.9]` has the sample 0.5, since no sample of a step of length 1 need fall in the interval `(0.49, 0.51)`. The terminal crossing at a step end that is a node runs with the fixed step 0.1 and the step ends as nodes, since with a dyadic step `ntrp1` at `theta = 1` equals the step end bit for bit. In the restarted `'left'` call on `y[0] - 10` the search of the first call meets an exact zero on the interpolant and returns it, so the restarted call starts on the surface and its event is the descent; the test asserts that case. The rule that a `'left'` root is never the start of the step is tested by a condition of `t` that jumps between the float 0.3 and the next, whose restarted call reports the float after 0.3, and by `find_root` directly. A component that crosses after `te` inside the bracket of `te` is tested to cost the probe only, by counting the condition calls, and the nudge, whose stored components I6b sets, by the brackets of one step with the stored state set in the test. I6b: a `ContinuousCallback` with an `affect` reversing `0.9` of the velocity (`rootfind='left'`) reproduces the ten bounces in one call within `rtol=1e-5` and reports each bounce once; two callbacks whose components cross at the same `te`, one acting and one record-only, are both handled at `te`, the record-only one logged and the modification protocol run once; a grid run with an acting event for `ImplicitEuler`, `Trapezoid` and `Rodas3`, whose interpolants read the step's end state: every row at a node `< te` byte-equal to the same run without the event. Recorded at I6b: `ImplicitEuler` and `Trapezoid` ship with I7a, so the grid run takes `Rodas3` and a `Rosenbrock` subclass with the default linear interpolant of `Algorithm`, which is the interpolant of both, and I7a adds the two to its parameters. Recorded at I7a: at `rtol=1e-6` their steps are shorter than the spacing of the grid, so the grid also gets a node halfway between the start of the crossing step and `te`, both taken from runs without a grid, since nodes change neither the steps nor `te`; a further test checks that a `'left'` crossing whose affect leaves the state unchanged is reported once, which needs the components stored by the protocol for the nudge |
| `test_callbacks.py` | I6b | `preset_time_callback` at 0.5 that changes a parameter of an algebraic equation: rows `(0.5, left)` and `(0.5, right)` with `save_positions=(True, True)`, the algebraic residual after the event below `1e-6` (`DaeIc`); `save_positions=(False, True)` gives one row at 0.5 on a grid that does not contain 0.5, and two rows when every step is saved, as in SciML; a change of `dae.M.data` at a `tstop`, as `ModeSwitch` does, is used by the next step; the same change between two `step()` calls followed by `model_modified()` is used by the next step and `interp` then accepts only `t`; an affect that calls `terminate()`, and `terminate()` between two `step()` calls; a discrete callback without `tstops`; after a modification protocol `cache.superlu` is empty; `DaeIc` failure after a callback gives `ret == 'failed'` without raising. Recorded at I6b: the file also checks that a KLU analysis with a matching is dropped by the protocol and one without is kept, that an interpolation of Rodas3 between two `step()` calls after an affect at the step end is byte-equal to the run without it, and the claim of Section 8.4 that a run changed at 0.5 is byte-equal after 0.5 to two calls split there, for `Rodas3` and `Rodas4` with `fix_h` on both backends, with and without the matching |
| `test_services.py` | I7a | `implicit` solves `M y - h*gamma*F(t, y) = rhs` to the Newton tolerance on `vdp` and `dae_test`; `slope=True` returns `k` within the Newton tolerance of `F(t, y)`; `W(gamma)` factorizes once per `gamma` per attempt; `s.F0`, `s.J0` and `s.dFdt()` are evaluated once per step and kept on retries; `s.D` marks the algebraic rows, explicit zeros of `M.data` included; `s.f` equals `M^-1 F` on model E and raises for a model with algebraic equations; `out=` of `F`, `f`, `W(...).solve` and `implicit` returns results byte-equal to the out-of-place call and allocates no array for `F` and `f`; the counters. Recorded at I7a: the file also checks the three `StepFailure` messages of the Newton iteration, and `ImplicitEuler` and `Trapezoid`, namely their orders on A by the largest error over the steps (Section 13), a run from A_delta at `atol = 1e-10` that succeeds while the same method without `D` fails, and a run on `vdp`; the allocation is measured with `tracemalloc`, which sees NumPy's arrays |
| `test_author_contract.py` | I7b | `check_algorithm` passes for `Rodas3`, `Rodas4`, `Rodasp`, `Rodas5P`, `ImplicitEuler` and `Trapezoid`; `TypeError` for a class without `perform_step`, for a `perform_step` whose signature does not match `inplace`, for an adaptive formula algorithm that returns no error, for a non-adaptive one that returns an error without `fix_h`, for an in-place adaptive one that sets no `EEst`, for a wrong shape of `y` or `err`, and for an explicit algorithm on a DAE; a toy formula subclass of an in-place toy class uses the formula dispatch and the parent's hooks; a table-only subclass built with `RosenbrockTableau.from_hairer` from the `rodas4` tables of `param.py` gives trajectories byte-equal to `Rodas4`; `check_algorithm` on an explicit toy algorithm, a fixed-step toy algorithm, and a toy algorithm that keeps stale data across a modification, which the history check must reject. Recorded at I7b: the stale toy is the trapezoidal rule that takes `f0` from the end slope of the last accepted step, first same as last; a Newton starting value kept across the change leaves no trace on the linear model A, where the simplified Newton with the exact `W` refines both starts to the same bits. Each check is also shown rejecting an algorithm broken in the way it guards against, which passes every earlier check; the rendered check with one that scales its error when the Jacobian has 32-bit indices, as a rendered one has. The checks `Opt` and `y0` run alone through the private `_check(alg, names)`: the contract gives no algorithm access to the caller's `Opt` or `y0`, so `y0` fails only with a core that writes the initial state, and the state across calls that breaks the repeatability of `Opt` meets `saveat` first. `test_events.py` gains the `i7b` test of Section 11.4 |
| `test_deprecation.py` | I8 | calling legacy `Rodas` raises one `DeprecationWarning` whose message starts with `Rodas is deprecated` and whose `filename` and `lineno` are the test's call line, both through the `dae_io_parser` wrapper and through `Rodas.__wrapped__`; `Rodas4()` raises none. Recorded at I8: the file also checks that Python's default filters show the warning to a caller in `__main__` and hide it from any other module, which is what the computed stack level serves, that every call warns, and that the filter of `pyproject.toml` hides the message |
| `test_api.py` | I8 | `solve(...)`, `init(...).solve()`, a `step()` loop and `Rodas4()(...)` byte-equal; `Vars` in gives `TimeVars` `Y` and `ye`; `Solverz.integrator.__all__` equals the list of Section 3.2; `from Solverz import Rodas4, ContinuousCallback` works. Recorded at I8: the four entries are compared for `Rodas4()` and `alg=None`, `Rodas3`, `Rodas4(legacy_compat=True)`, `ImplicitEuler` and `Trapezoid` on `[t0, tend]` and a grid, with a terminal `opt.event` in both configurations, and, without the legacy-shaped call, which takes no callbacks, with a `ContinuousCallback` that has an `affect`; `Stats` counters are compared too, and `solve`, `init` and `Integrator` are asserted absent from the top level |

The out-of-place `Rodas4` and the in-place `ImplicitEuler` and `Trapezoid` of D7 are not in the repository; their bit identity with the shipped classes is asserted by the benchmark before it times them (Section 17.1).

Every milestone gate runs the whole Solverz suite on the server twice, once with the default backend and once with `SOLVERZ_LINSOLVER=superlu`, through `server_test.sh`, which forwards that variable from the tooling step before I1 (Section 18).

## 16. Downstream gate: SolPSDyn

### 16.1 Tree and environment

SolPSDyn `c16a041` is the newest commit that imports on Solverz `056e87a`: its child `e7d2f36` makes `SolPSDyn/dae/__init__.py` import `device.py`, which needs `Solverz.sym_algebra.intermediate`, absent at `056e87a`. `event_loop.py`, `test_event_loop_nodes.py` and `test_switched_wrapper.py` are unchanged between `c16a041` and the SolPSDyn head, and a private benchmark record gives `c16a041` as the SolPSDyn state of the C0 baselines.

1. Pinned trees. SolPSDyn at `c16a041`, and SolAlg, SolMuseum and SolUtil at the commits of 2026-09-30 that the operations notes record, each copied to the server as a tree of that commit. If one of them fails to import on `056e87a`, the newest ancestor that imports is taken and recorded, as for SolPSDyn. The version of `andes` in the server environment is recorded. Nothing is committed to any of these repositories.
2. On the server, `PYTHONPATH` lists the integrator worktree, the four trees and the gate's tooling before the environment's own packages. The runner asserts that `Solverz.__file__`, `SolPSDyn.__file__`, `SolAlg.__file__`, `SolMuseum.__file__` and `SolUtil.__file__` start with those directories, and that `SOLPSDYN_SKIP_HEAVY` is unset or `0`, since `test_ieeex1.py`'s parity test is skipped otherwise (`SolPSDyn@c16a041:SolPSDyn/dae/test/test_ieeex1.py:46`). Thread pools are pinned as in Section 17.2. On the laptop the editable installs point at other trees, among them a Solverz on `feat/kernel-eqn`, which is why the gate must not run there. The directories and commands are in the operations notes.
3. The pytest plugin `solpsdyn_gate`, kept outside the repository with the operations notes and loaded with `-p solpsdyn_gate`, reads `C2_GATE_MODE` in `{default, compat}`, replaces `SolPSDyn.dae.event_loop._default_solver` with a function returning the adapter, and in `pytest_collection_modifyitems` replaces the module global `Rodas` of each collected test module with the adapter:

```python
def kernel_rodas(dae, tspan, y0, opt=None):
    scheme = (opt or Opt()).scheme
    return Rosenbrock.from_scheme(scheme, legacy_compat=(MODE == 'compat'))(dae, tspan, y0, opt)
```

4. Files: `SolPSDyn/dae/test/test_event_loop_nodes.py`, `test_switched_wrapper.py`, `test_ieeex1.py` and `test_datacenter.py`, all at `c16a041`, each in both modes. `test_switched_wrapper.py` is the only gated file that exercises `opt.event` under `EventLoop`, with terminal stops, `te` repeated and `T[-1] == te`; its `pytest.importorskip('SolAlg')` at line 55, and a skip of `test_ieeex1.py`'s parity test, count as gate failures, not as skips. The report lists, per file and mode, the numbers of passed, failed and skipped tests, and for `test_switched_wrapper.py` the `n_suppressed` count of `EventLoop`'s row events per run (`SolPSDyn@c16a041:SolPSDyn/dae/event_loop.py:188, :227`).

### 16.2 Expected differences and their causes

| Assertion | Expectation | Cause |
|---|---|---|
| `test_event_loop_nodes.py:59-66`, a two-node loop run byte-equal to a bare run | pass | both sides run the kernel; no call depends on an earlier one (F1) |
| `test_event_loop_nodes.py:71-157`, grid within `1e-12`, accuracy `1e-7` and `1e-9`, `hmax` independence, event rows, exact event instant | pass | `T[-1] == te` exactly and nodes exact |
| `test_switched_wrapper.py:170`, pinned rows exactly on the bound in every segment's last row | may fail | a Rosenbrock stage solve keeps the linear algebraic row `bound - x = 0` exactly only if the LU solve returns an exact zero there; a failure is attributed to the method |
| `test_switched_wrapper.py:249-256`, loop accepted steps fewer than the reference solver's steps on the unswitched model | may fail | the reference is the implicit solver that the test imports from SolAlg, called directly (lines 249-256 and 359-365), a different method |
| `test_switched_wrapper.py:187, :200`, bound respected with zero crossings, exactly and without a tolerance | may fail | with `rootfind='right'` the stopped state lies on the crossed side, and the assertions rely on `EventLoop` projecting every stopped row; a row that the per-row dwell holds is emitted unprojected, below the bound (`SolPSDyn@c16a041:SolPSDyn/dae/limiter.py:599-601, :646-647`, `event_loop.py:322-333`). A failure is attributed to this cause when the report's `n_suppressed` is positive for that run |
| `test_switched_wrapper.py:214-217`, the `hmax` ladder | pass expected | the kernel never discards a crossing near a step start and reports every simultaneous component |
| `test_ieeex1.py:342-350`, one `Opt` with `hmax=None` reused across three calls | pass within the test's tolerances expected; step counts differ | legacy writes `hmax` = the first span into the shared `Opt` and caps later calls; the kernel does not |
| `test_datacenter.py:162, :193, :201` at `c16a041`, dense-output accuracy with fresh `Opt` | pass | no event, accuracy checks only |
| any assertion on `nfeval`, `ndecomp` or `nsolve`, or on `EventLoop`'s own statistics | differs | Section 5.9 counts every call |
| a test that expects `ValueError('Need Better y0')` to propagate | differs | a `DaeIc` failure now ends the run with `ret == 'failed'`; `EventLoop` then raises `IntegrationFailure` at `t0`, and a private study script classifies such a run as a step-size collapse, not as a raised exception. No gated test is known to expect it |

A failure outside the "may fail" rows, or a "may fail" row whose cause is shown to be the kernel, blocks milestone I9. The report lists each test and mode with pass, fail, skip and the attributed cause.

## 17. Benchmarks (D7)

### 17.1 Scripts and models

The benchmark scripts are new scripts beside the existing benchmarks of the SDCIB model in a private benchmark record, and run as those do; the operations notes give their location. The style counterparts of D7 live outside the repository, as D7 and plan revision item 5 require.

- The style counterparts:
  - `FormulaRosenbrock(scheme)`, a subclass of `Rosenbrock` with `inplace = False` and the tableau `Rodas_param(scheme)`. Its `perform_step(self, s)` writes the step of Section 10.3 out of place: `K = np.zeros((s.n, st))` per attempt; `dfdt0 = s.h * s.dFdt()`; `rhs = s.F0 + g[0] * dfdt0`; `W = s.W(gamma)`; `K[:, 0] = W.solve(rhs)`; per stage `sum_1 = K @ alpha[:, j]`, `sum_2 = K @ gammatilde[:, j]`, `y1 = s.y0 + s.h * sum_1`, `rhs = (s.F(s.t + s.h * a[j], y1) + s.M @ sum_2) + g[j] * dfdt0`, `K[:, j] = W.solve(rhs) - sum_2`; `sum_1 = K @ (s.h * b)`, `y = s.y0 + sum_1`, `err = sum_1 - K @ (s.h * bd)`; `s.cache.K[...] = K` for the interpolant; it returns `(y, err)`, and `(y, None)` when `not s.adaptive`. The shipped class returns its error through `error_norm` on the same vector, so both styles make the same service calls and the same counted residuals. It inherits `alloc`, `addsteps`, `interpolant` and `controller`.
  - `InplaceImplicitEuler` and `InplaceTrapezoid`, subclasses with `inplace = True` whose `perform_step(self, integ, cache)` computes the formula of Sections 12.5 and 12.6 with the same association into buffers from `alloc`: for `ImplicitEuler`, `integ.implicit(t + dt, 1.0, M @ y0, out=integ.u)`; `np.subtract(integ.u, y0, out=c.a)`; `Ma = M @ c.a`; `np.multiply(D, integ.F0(), out=c.b)`; `np.multiply(c.b, dt, out=c.b)`; `np.subtract(Ma, c.b, out=c.b)`; `np.multiply(c.b, 0.5, out=c.b)`; `integ.W(1.0).solve(c.b, out=c.e)`; `integ.EEst = integ.error_norm(c.e)`. `InplaceTrapezoid` is written the same way.
- `bench_core_e2e.py`: legacy `Rodas`, the kernel in the legacy-compatible configuration and the kernel in the default configuration, back to back. Models: rendered `alloc(n)` for `n` in 10, 100, 1000, 10000 with `jit=True` on `[0, 1]` with `Opt(rtol=1e-6, atol=1e-8, hmax=1e-2)`, as the existing allocation benchmark of the private record runs it; rendered `ladder(n)` for `n` in 28, 1000 and 10000, whose `W` is not diagonal; the SDCIB model at `n = 28`, inline and rendered, `rtol` 1e-4, 1e-6 and 1e-8 over 2 s, as a private benchmark record runs it, if it runs (17.3); and one variant of each model on an output grid of 2001 nodes. Before timing, the script asserts that the legacy-compatible trajectories are byte-equal to legacy, and it reports `max|dY|` of the default configuration against legacy on a 201-node grid.
- `bench_core_styles.py`: the built-in in-place `Rodas4` against `FormulaRosenbrock('rodas4')`, and `ImplicitEuler` and `Trapezoid` against `InplaceImplicitEuler` and `InplaceTrapezoid`, in the default configuration, on the models of `bench_core_e2e.py`. Before timing, the script asserts byte-equal `T`, `Y` and equal counters, for the four Rosenbrock schemes, both configurations, `[t0, tend]` and a grid.
- `bench_core_events.py`: `ball` and `orbit` with `Opt(event=...)`, and `EventLoop` in root mode on the reproducer of `SolPSDyn@c16a041:SolPSDyn/dae/test/test_switched_wrapper.py`, `_build(32, switched=True)`, over its `TEND`; legacy, compatible and default back to back. It reports the total wall clock, the cost per accepted step and per `EventLoop` segment, `ncondition`, and the number of interpolations.
- `bench_core_calls.py`: 1000 calls of one to three steps each on the same rendered `alloc(28)`, legacy against the kernel, which measures the setup cost of a call that `EventLoop` pays per segment.
- `bench_core_null.py`: a model with `n = 2` whose `F` and `J` return constants, on which the model and the linear algebra cost almost nothing, so that the overhead of the loop per attempt appears directly in microseconds, legacy against both configurations.
- `bench_core_parity.py`: if SDCIB runs, the lockstep of D3a over the first 12 attempts of an SDCIB run (`t = 0`, `dt0 = 5e-8`, `rtol = 1e-8`, `atol = 1e-10`, as in a private benchmark record), printing the maximum difference of `ynew` and `err` per step, expected `0.00e+00` at all 12.

### 17.2 Protocol and metrics

- Server only (D8), in the directory and environment of the operations notes: `PYTHONPATH` set to the synced worktree and the benchmark tooling, a per-worktree `NUMBA_CACHE_DIR`, one process pinned to one core, and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMBA_NUM_THREADS` all set to 1, so that no thread pool oversubscribes the pinned core.
- One warm-up run of each variant first, so that Numba compilation of the rendered module and of `ntrp1` and `ntrp2` is excluded; then three repetitions in the order legacy, compatible, default, repeated.
- Per variant: total wall clock (minimum of three and the spread, maximum minus minimum); accepted and rejected steps; `nJeval`, `ndecomp`, `nsolve`; the true residual count, taken from `Stats` for the kernel and from a counting wrapper in a separate untimed run for legacy, whose `nfeval` omits calls (Section 5.9); the number of KLU analyses, counted by wrapping `KLUSymbolic.__init__` in the harness; milliseconds per accepted step and per attempt.
- Time attribution, in the timed runs of every variant alike, by wrappers installed in the harness only and never in production code: `perf_counter` accumulators around `dae.F`, `dae.J`, `lu_decomposition` and the `solve` of the factorization it returns. The report gives the time in each and the remainder, `wall - sum`, which is the overhead of the loop and of the SciPy chain of Section 8.2. The chain is the same code in legacy and in the kernel, so the difference of the remainders is the difference of loop overhead. The remainder is reported per attempt. "Per `F` call" is the time measured inside `F` divided by the number of calls, and the cost of one call through the core's `F` service, the counting wrapper plus, in the formula style, the allocation of the result, is measured separately, as the existing allocation benchmark of the private record measures the cost of one `F` call.
- Kernel against legacy per-step overhead: compatible against legacy runs the same step sequence but makes `s + 1` residual calls on a first attempt and `s - 1` on a retry against legacy's `s + 2`, reuses `dF/dt` across retries and allocates no temporaries per attempt, so its wall-clock difference is overhead minus savings; the time attribution separates the two. Default against legacy is reported per accepted step and in total, with the step counts, so that a difference is attributed to a more expensive step or to more steps (plan:201-203). The null model gives the overhead per attempt without either saving.
- Compilation: the core changes no printer, so the rendered module's compile time is unchanged; the report states this, and states any change measured.
- What the style comparison measures: for the Rosenbrock family, the in-place and the out-of-place step differ in every operation of Section 10.3. For `ImplicitEuler` and `Trapezoid`, both styles call the same allocation-free Newton of Section 12.4, so the comparison measures the code of `perform_step` around it; the report states this.

### 17.3 SDCIB availability

The current SDCIB model of a private benchmark record passes `kernels=` to `dev.mdl` and `infinite_bus_closure` and needs the SolPSDyn head, which needs Solverz `feat/kernel-eqn`. The benchmark copies it with the two `kernels=` arguments removed and imports SolPSDyn from the `c16a041` tree of Section 16. If that fails to build or to integrate, SDCIB is reported as not run with the error, and `ladder(28)` and `alloc(28)` stand in for the per-step overhead at `n = 28`, stated as stand-ins: `alloc(28)` has a diagonal Jacobian and `ladder(28)` a tridiagonal one, and both are autonomous, unlike SDCIB. Note that the SDCIB build of the `c16a041` era has 105 Jacobian entries and the later named build 96 (a private benchmark record), so baselines from different eras do not describe the same model.

### 17.4 Reporting rule

From plan:197-205. Any metric of the new implementation that is slower than legacy by more than 2 percent or by more than the spread of the three repetitions, whichever is larger, is marked "performance regression" in the milestone report and in the private benchmark records, with the numbers and the attribution to cost per step, number of steps, or, for events, number of condition calls. Nothing is accepted silently; the user decides. The same rule marks the formula style against the in-place style. The event benchmarks fall under the rule too: if the nine condition calls per step of the adapter (Section 11.6) are flagged on `EventLoop` runs, the user decides whether the adapter keeps `interp_points = 10`.

## 18. Milestones

Tooling before I1, with no commit on `feat/integrator-core`: `c2-tools/server_test.sh` forwards `SOLVERZ_LINSOLVER` to the server when the caller has set it and does nothing otherwise, so that every gate can run the suite twice from I1 on; `c2-tools` and the scripts of Section 17 are synced to the server directory of the operations notes with `rsync -az`, as `server_test.sh` syncs the worktree.

Milestones I1 to I8 are each one implementer's work, end with their gate green on the server (`../c2-tools/server_test.sh <worktree> [args]`, with the default backend and with `SOLVERZ_LINSOLVER=superlu`), and are each one conventional commit on `feat/integrator-core`. I9 and I10 produce reports and no commit on that branch. The stop for the user's confirmation that the design document requires applies to its milestones C0 to C6, so C2 as a whole stops for review after I10; I1 to I10 run without a stop in between. The pull request is squash-merged into one commit upstream.

| Milestone | Content | Gate |
|---|---|---|
| I1 | package skeleton and `__all__`; `options.py`; `algorithm.py` (`Algorithm`, `StepFailure`, `StepContext` stub); `saving.py` | `test_matmul_out.py` byte-equal on the server; `test_import.py` |
| I2 | `linalg.py` with the chain, the backends and `solve(b, out)`; `klu_decomposition.solve_into`; `derivative.py`; `tests/integrator/models.py`; `legacy_rodas.py` | `test_legacy_transcription.py`; the I2 tests of `test_linalg.py` and `test_derivative.py` |
| I3 | `rosenbrock.py`: tableau, cache, `perform_step`, both norms, `ntrp1` and `ntrp2` interpolation; enough of `Integrator` to run one attempt | `test_rosenbrock_lockstep.py`, every attempt byte-equal, both backends |
| I4 | the loop of Section 5; `policies.py` with `LegacyRodasPolicy`; `controllers.py` with all four controllers; `solve`, `init`, `step`, `__call__`, `Vars`; `DaeIc` and its guard; the failure path; `Stats` | `test_compat_parity.py`, `test_controllers.py`; the I4 tests of `test_failure_and_opt.py`, `test_derivative.py` and `test_dae_xlsx.py` |
| I5 | `DefaultPolicy`: `'ode23s'`, the default norm, `tstops`, saving, the cache rule of Section 8.4, the dense factorization, the Rodas3 pairing | `test_saveat_tstops.py`, `test_rendered_inline.py`; the I5 tests of `test_linalg.py`, `test_failure_and_opt.py` and `test_dae_xlsx.py` |
| I6a | `callbacks.py`: continuous callbacks, detection with the skip rule, `find_root`, the `opt.event` adapter, handling of terminal and recording events | the I6a tests of `test_events.py` |
| I6b | affects, the modification protocol, `model_modified`, discrete callbacks, `preset_time_callback` | the I6b tests of `test_events.py`, `test_callbacks.py` |
| I7a | `StepContext` and its services, `nlsolve.py`, `ImplicitEuler`, `Trapezoid` | `test_services.py` |
| I7b | `testing.py` | `test_author_contract.py` |
| I8 | the warning of Section 14; `pyproject.toml` filter; top-level exports; docs: `docs/src/integrator.md`, with the conditions of D3b parity, the cost of events and the exact-zero start of Section 11.3; `docs/src/integrator_adding_algorithms.md` (the contract table, `ImplicitEuler` in full, the services, the history rule, `check_algorithm`, and the statement that in C2 only the Rosenbrock family is table-driven); `docs/src/index.rst` toctree; `docs/src/reference/index.rst` entries for the public names; the examples in `docs/src/intro.md:24, :46` and `docs/src/gettingstart.md:279, :333` switched to `Rodas4()`; the release note of Section 18.1; this document, under the publication rule of Section 0 | `test_deprecation.py`, `test_api.py`; the whole Solverz suite green on the server with both backends |
| I9 | the SolPSDyn gate of Section 16 in both modes | the report `c2-tools/reports/I9_solpsdyn_gate.md`, with every failure attributed and no unattributed failure |
| I10 | the benchmarks of Section 17 | the report `c2-tools/reports/I10_benchmarks.md` and the sections in the private benchmark records, with regressions marked; the user decides |

Recorded at I1. `Algorithm` has every member of Section 12.1 from I1 on except `controller` and `__call__`, which need `controllers.py` and the Integrator and are added with I4. The style check of Section 5.2, step 1, is `check_style(alg)` in `algorithm.py`, cached per class and per value of `inplace`, and the Integrator calls it from I4 on. The `StepContext` stub holds `n`, the read-only view `y0`, and the members that read a field of the Integrator, namely `t`, `h`, `new_step`, `M`, `p`, `rtol`, `atol`, `adaptive` and `cache`; the services of Section 12.2 are added with I7a.

Recorded at I3. The Integrator exists from I3 on and calls `check_style` from I3 on. I3 also implements the dispatch of both styles of Section 12.1, which sets `EEst = None` before the call in both styles, `Integrator.interp` of Section 10.5, `Rosenbrock.from_scheme` and `RosenbrockTableau.from_hairer`, and the Rodas3 pairing that the table lists with I5 (Section 10.5). `__all__` gains `Integrator` and the six names of `rosenbrock.py`; `solve` and `init` follow with I4.

Recorded at I8. The example of `README.md` is the example of `docs/src/intro.md` and is switched to `Rodas4()` with it, although the row above does not list the file, since a first example that warns would contradict the deprecation. The two new pages form a toctree of their own, captioned "Solvers", after "Start here"; the introduction of `docs/src/index.rst` names the integrator, and the entry of legacy `Rodas` in `docs/src/reference/index.rst` gains one sentence that points at the integrator. The release note is placed inside 0.11.2 as O2 decides, since 0.11.2 was not released on 2026-09-30, the upstream tags ending at 0.11.1; it adds the sections New, Deprecated and "Differences from the legacy `Rodas`" to that version. The moved material of the publication rule, with the unpublished text of Sections 16.1 and 17 and of the tooling paragraph, is in the operations notes. The documentation was not built with Sphinx, since the server environment has no Sphinx and the gate does not include a documentation build; its examples were run on the server.

### 18.1 Release note

The note describes the core, the deprecation, and the caller-visible differences of the default configuration listed in Appendix B, among them that a `DaeIc` failure ends a run with `ret == 'failed'` instead of raising, that `hinit <= 0` raises `ValueError`, that a non-increasing grid raises `ValueError`, that a non-terminal event no longer shortens the step or adds a row, and that `ret` is set on success. Where it goes depends on O2 (Section 19). The version is chosen at release time; a 0.12 release crosses SolMuseum's cap `Solverz>=0.10.0,<0.12` and then needs SolMuseum released first (`CLAUDE.md`, release process).

## 19. Decisions on the open items

Decided by the project lead on 2026-09-30.

O1. F1 and F2 of Section 2.2 are accepted.

O2. Implementation proceeds on `feat/integrator-core` at `056e87a`. The #188 branch carries `26673bd` and a correction of its 0.11.2 release note, which now states the measured step count instead of a general bit-for-bit claim; both are pushed once the cookbook fix that its CI depends on is merged. After #188 merges, the C2 branch is rebased, the cited line numbers of `tests/test_rendered_F_aliasing.py` are updated, and the C2 release note goes inside 0.11.2 if that version is unreleased, and under a new heading otherwise.

O3. The benchmarks try the `c16a041` copy of `sdcib.py` with the `kernels=` arguments removed; if it fails, `ladder(28)` and `alloc(28)` are reported as stated stand-ins.

## Appendix A. Correspondence with OrdinaryDiffEq.jl

| OrdinaryDiffEq.jl at `7393799` | Core |
|---|---|
| `solve!`, `lib/OrdinaryDiffEqCore/src/solve.jl:1062-1104` | `Integrator.solve`, `Integrator.step` |
| `loopheader!`, `integrators/integrator_utils.jl:84-127` | `loopheader` |
| `apply_step!`, `:175-203` | `apply_step`, the commit point |
| `handle_step_rejection!`, `:130-150` | the rejection branch of `loopheader` |
| `modify_dt_for_tstops!`, `:271-327`; `handle_tstop_step!`, `:329-336` | `policy.modify_dt_for_tstops`, `_skip_to_tstop` |
| `_savevalues!`, `:343-432` | `policy.savevalues` |
| `_postamble!`, `:494-590` | `postamble` |
| `_loopfooter!`, `:600-680` | `loopfooter` |
| `handle_callbacks!`, `:1097-1146` | `handle_callbacks` |
| `fix_dt_at_bounds!`, `:1263-1276`; `calc_dt_propose!`, `:1213-1230` | `policy.fix_dt_at_bounds`, `policy.dt_propose` |
| `handle_tstop!`, `:1310-1334` | `policy.handle_tstop` |
| `IController`, `PIController`, `integrators/controllers.jl:659-698, :754-843` | `controllers.py` |
| `find_callback_time`, `nudge_tprev`, `check_event_occurrence`, `find_root`, `lib/DiffEqBase/src/callbacks.jl:473-603` | `callbacks.py` |
| `apply_callback!`, `apply_discrete_callback!`, `callbacks.jl:668-796` | `handle_callbacks` |
| `change_t_via_interpolation!`, `reeval_internals_due_to_modification!`, `integrators/integrator_interface.jl:40-91` | the event branch of Section 11.5 and the modification protocol, Section 11.7 |
| `u_modified!` | `Integrator.model_modified` |
| `calc_W!`, `do_newJW`, `lib/OrdinaryDiffEqDifferentiation/src/derivative_utils.jl:541-563, :826-959` | `IterationMatrix`, `integ.J0`, `integ.W` |
| `calc_tderivative!`, `derivative_utils.jl:181-227` | `integ.dFdt` |
| Rosenbrock `perform_step!`, `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl:734-852` | `Rosenbrock.perform_step`, kept in Solverz's `K` form for parity |

Differences kept on purpose: the core writes each algorithm once, not as a constant cache and a mutable cache (`docs/src/devtools/contributing/adding_algorithms.md:27-47` in OrdinaryDiffEq.jl); the Rosenbrock uses `W = M - dt*gamma*J` and the `K` form of `param.py`, not `J - M/(dt*gamma)` and the `(A, C)` tables; the default controller for Rosenbrock is the `IController` with legacy constants, not SciML's `PIController` with `qsteady_max = 6/5` and `qmax_first_step = 10000` (`lib/OrdinaryDiffEqCore/src/alg_utils.jl:740-742, :863-869, :880`); after an event the core keeps the computed step for interpolation instead of recomputing it over the shortened interval.

## Appendix B. Legacy Rodas behaviour, reproduced or not

Reproduced by the legacy-compatible configuration on event-free runs, under the conditions of Section 6: the step of Section 10.3; `J` only on the first attempt of a step; `dF/dt` by the legacy formula; the legacy error norm with the `1e6` override; the controller with its `facmax` state seeded from `opt.facmax`; `hmin = 16*spacing(t0)`; the stretch rule and `0.5*(tend - t)` cap; `t = t + dt`; the absolute termination test; the initial step; the fresh KLU cache; the per-stage dense solve; saved rows through `ntrp1` and `ntrp2`; `DaeIc` at entry.

Not reproduced, in either configuration: the write-back of `hmax` and `facmax` into `Opt` (`rodas.py:77, :349, :354`); the silent stop on a factorization error (`:186-189`); the uncaught `klu_solve` error (`:191, :200`) and the uncaught arithmetic error of a rendered residual; the unset `succeed` (`Solverz/solvers/stats.py:12`) and the `ret` of `None` on success, which is now `'success'` or `'terminated'`; the printed `NaN` warning (`:212`); the continued attempts with a `NaN` step until the rejection limit (`:355`), which end at the next attempt instead, with the same prefix; the buffers of 10001 rows and their overflow (`:82-84, :106-108, :340-344`); the views returned as `T` and `Y` (`:357-358`); the float `ie` (`:108`); the uncounted residual calls (`:375-380, :404-405`); the `TypeError` of `fix_h` without `hinit` (`:152, :162`), which is a `ValueError` before any step; the clamp of `hinit <= 0` to `hmin` (`:111-117`), which is a `ValueError`; the propagated `ValueError('Need Better y0')` of `DaeIc` (`:85`), which ends the run with `ret == 'failed'`, `T == [t0]` and no raise; the failure at `t0 == tend` (`:135-138`), which returns one row with `'success'`; and the event semantics of `:228-299`, namely the strict sign test that misses an exact zero, the secant start followed by bisection whose returned `te` is an unevaluated midpoint while `ye` belongs to the previous probe, the exclusion of crossings within `event_duration` of the step start that also skips every later component of the step and leaves the next step's reference value at an interior point, the shortening of the step at non-terminal events, which also adds a `(te, ye)` row to `T` in 2-node mode, and the stop at the first terminal component in index order.

Not reproduced in the default configuration only: the silent end of saving at the first repeated node of a non-increasing grid, which is a `ValueError`; the scaling of the error by `|ynew|` alone; `dF/dt` with `ddt = 1.49e-16` at `t = 0`; the fresh KLU analysis in every call below the matching threshold; and `t = t + dt` at the end of the span, which is `t = tend` exactly.
