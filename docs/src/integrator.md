(integrator)=

# Integrating a DAE

`Solverz.integrator` integrates the numerical DAE $M y' = F(t, y, p)$ that `made_numerical` and `module_printer` produce. One stepping loop owns everything that does not depend on the method: step-size control, acceptance and rejection, the output grid, stop times, events and callbacks, the linear algebra and the statistics. A method only states how one step is taken. The loop follows the division of work of [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), and its names follow SciML's, so its behaviour can be compared with that package line by line.

The methods are

| Class | Method | Order | Interpolant |
|---|---|---|---|
| `Rodas4()` | Rodas4 of Hairer and Wanner, the default | 4 | the dense output of the method |
| `Rodas3()` | Rodas3 | 3 | cubic Hermite interpolant |
| `Rodasp()` | Rodasp of Steinebach | 4 | the dense output of the method |
| `Rodas5P()` | Rodas5P of Steinebach | 5 | the dense output of the method |
| `ImplicitEuler()` | backward Euler | 1 | linear |
| `Trapezoid()` | trapezoidal rule | 2 | linear |

The four Rosenbrock methods use the coefficient tables of the legacy `Rodas` solver. `ImplicitEuler` and `Trapezoid` are written as formulas in a few lines each; they are the templates of {ref}`adding an algorithm <integrator_adding_algorithms>`, which explains how a new method is written and checked.

## A first example

The apple of the {ref}`introductory example <intro>`, launched upwards at 20 m/s, stops when it hits the ground:

```python
import numpy as np
from Solverz import Model, Var, Ode, Opt, made_numerical, Rodas4

m = Model()
m.h = Var('h', 0)
m.v = Var('v', 20)
m.f1 = Ode('f1', f=m.v, diff_var=m.h)
m.f2 = Ode('f2', f=-9.8, diff_var=m.v)
bball, y0 = m.create_instance()
nbball = made_numerical(bball, y0, sparse=True)


def events(t, y):
    value = np.array([y[0]])
    isterminal = np.array([1])
    direction = np.array([-1])
    return value, isterminal, direction


sol = Rodas4()(nbball, np.linspace(0, 30, 100), y0, Opt(event=events))
print(sol.te, sol.T[-1])   # both about 40/9.8 = 4.0816
```

The instance `Rodas4()` selects the method, and calling it has the form of the legacy `Rodas(dae, tspan, y0, opt)`. Because `y0` is a `Vars`, `sol.Y` is a `TimeVars`, and `sol.Y['h']` is the height at every saved time.

## Three ways to call

```python
from Solverz.integrator import solve, init, Rodas4

sol = Rodas4()(dae, tspan, y0, opt)                      # the legacy form
sol = solve(dae, tspan, y0, alg=Rodas4(), opt=opt,       # with callbacks and stop times
            callbacks=(), tstops=())

integ = init(dae, tspan, y0, alg=Rodas4(), opt=opt)      # one step at a time
while integ.step():
    print(integ.t, integ.u)
sol = integ.solve()
```

- `solve(dae, tspan, y0, alg=None, opt=None, *, callbacks=(), tstops=())` integrates and returns the result. `alg=None` means `Rodas4()` and `opt=None` means `Opt()`.
- `alg(dae, tspan, y0, opt=None)` is `solve(dae, tspan, y0, alg=alg, opt=opt)`. It is the form that code written for the legacy `Rodas` calls, such as an event loop that passes its solver as `solver=Rodas4()`.
- `init(...)` returns the `Integrator` before its first step. `integ.step()` advances by one accepted step and returns `False` once the run is over. Between two calls, `integ.tprev` and `integ.uprev` are the start and `integ.t` and `integ.u` the end of the step just taken, and `integ.interp(tq)` evaluates the state at any `tq` inside it. When `integ.step()` returns `False` because the run failed, `integ.u` is the state at `integ.t`, the end of the last accepted step, and `integ.interp` accepts only `tq == integ.t`. `integ.solve()` runs to the end and returns the result, `integ.terminate()` ends the run, and `integ.model_modified()` is described in {ref}`changing the model <integrator_modification>`.

The three forms run the same code and return byte-identical results. `solve`, `init` and `Integrator` are imported from `Solverz.integrator`. The top level of Solverz exports only the methods `Rodas3`, `Rodas4`, `Rodasp` and `Rodas5P` and the classes `ContinuousCallback` and `DiscreteCallback`, so that `from Solverz import *` after `from sympy import *` keeps SymPy's `solve`. `ImplicitEuler`, `Trapezoid` and the other names of the package are imported from `Solverz.integrator`.

## Time span, output grid and stop times

- `tspan = [t0, tend]` saves every accepted step.
- A `tspan` of more than two entries saves the state at `tspan[1:]` by interpolation. The nodes do not change the steps that are taken, and they must increase strictly, otherwise `ValueError`.
- `T[0]` is `t0` and `Y[0]` is the consistent initial state that `DaeIc` computes from `y0`. After a successful run `T[-1]` is `tend` exactly, and on a grid the last row is the state of the last step, not an interpolated value.
- `tstops` are times at which a step must end exactly, for example the instant at which an input changes. Values outside the open interval `(t0, tend)` are ignored. A step that was shortened only to meet a stop time does not make the next step shorter.
- `t0 > tend` raises `ValueError`, since integration backwards in time is not supported. `t0 == tend` returns the single row at `t0`.

`y0` is never written. A `Vars` `y0` gives `TimeVars` rows in `Y` and `ye`, and an ndarray `y0` gives ndarrays.

## Options

`Opt` is read once at the start of a call and never written.

| `Opt` field | Meaning |
|---|---|
| `rtol`, `atol` | the tolerances; `atol` may be an array with one entry per variable |
| `f_savety` | the safety factor of the step-size controller |
| `fac1` | the smallest step factor, so that a new step is at least `fac1` times the last one |
| `fac2` | the largest step factor after an accepted step |
| `facmax` | the largest step factor of the first attempt |
| `hinit` | the first step; with `fix_h`, the fixed step |
| `hmax` | the largest step; `abs(tend - t0)` when `None` |
| `fix_h` | a fixed step `hinit` without error control |
| `linsolver` | `'klu'` or `'superlu'` for this call, see {ref}`the linear solver <integrator_linsolver>` |
| `event` | a legacy event function, see {ref}`events <integrator_events>` |
| `pbar` | a progress bar |
| `profile` | print the wall-clock time of the call |

The method is chosen by the class, and `opt.scheme` is not read. When `opt.scheme` names another method than the class, for example `Rodas4()` with `Opt(scheme='rodas3')`, the call warns with a `UserWarning` that points at the caller; `Rosenbrock.from_scheme(opt.scheme)` returns the class that `opt.scheme` names. `opt.event_duration` is not read either.

Before any step, `ValueError` is raised for `t0 > tend`, for a `hinit` that is not positive, for a grid that does not increase strictly, and for `fix_h` without `hinit`.

The step-size controller is the integral controller `IController`, with the exponent `1/k` for an error estimate of order `k`. After a rejected attempt no attempt enlarges the step until the next step is accepted. The error of a Rosenbrock method is the largest component of `err / (atol + rtol * max(|u|, |uprev|))`; `ImplicitEuler` and `Trapezoid` take the root mean square of the same vector. The smallest step at time `t` is 16 units in the last place of `t`.

## The result

The result is the `daesol` of the legacy solvers, `sol.T`, `sol.Y`, `sol.te`, `sol.ye`, `sol.ie` and `sol.stats`.

- `T` is a new float64 array and `Y` a new C-contiguous array with one row per entry of `T`.
- `te`, `ye` and `ie` hold the recorded events, with `ie` of type int64, and are `None` when nothing was recorded.
- `stats.ret` is `'success'`, `'terminated'` after a terminal event or `integ.terminate()`, or `'failed'`, and `stats.succeed` is `stats.ret != 'failed'`. After a failure `stats.t_fail` is the time at which the run failed.
- `stats.nstep` counts accepted steps and `stats.nreject` rejected and failed attempts. `stats.nfeval` and `stats.nJeval` count every call of `F` and `J`, including those of `DaeIc`, of `dF/dt` and of the interpolation, `stats.ndecomp` and `stats.nsolve` every factorization and solve, and `stats.ncondition` every evaluation of a callback condition.

## Failures

A run never raises for a numerical failure. It prints one line and returns the rows saved so far with `stats.ret == 'failed'`. For `x' = x**3` from `x = 1`, whose solution blows up at `t = 0.5`, `Rodas4()(dae, [0, 1], y0)` prints

```text
rodas4: the step size 6.776494840217467e-16 is too small at t = 0.49994546317488403; the solution is returned up to t = 0.49994546317488403.
```

The run fails when the step size becomes too small or is not finite, after more than 100 consecutive rejected attempts, when a step of a fixed-step run fails or leaves a state that is not finite, and when `DaeIc` finds no consistent initial values, at `t0` or after a change of the model. In the last case at `t0` the result holds the single row `(t0, y0)`.

Inside a step, a failed factorization or solve, a Newton iteration that does not converge, and an arithmetic error of `F` or `J`, such as the `ZeroDivisionError` that a residual compiled by Numba raises on a division by zero, reject the attempt, and the step is retried with half its size. A non-finite error estimate rejects the attempt with the smallest step factor. Any other exception, for example a `TypeError` in the model or in a callback, is a programming error and propagates.

(integrator_events)=

## Events and callbacks

### Legacy events through `opt.event`

`opt.event` is a function `event(t, y) -> (value, isterminal, direction)` of arrays, as for the legacy `Rodas`. A component of `value` crosses when it goes from nonzero to zero or to the other sign; `direction` -1 accepts only crossings from positive values, +1 only those from negative values, and 0 both.

- Every crossing is recorded in `te`, `ye` and `ie`, and a crossing of a terminal component ends the run. The run stops at the earliest terminal crossing `te`, with `T[-1] == te` exactly and `Y[-1]` equal to the recorded `ye` row.
- Every component that crosses at `te` is recorded, with `te` repeated, and `ie` lists them in increasing order.
- A crossing of a non-terminal component is recorded without shortening the step and adds no row to `T`.
- The time of a crossing is the first float at which the component has crossed or is zero, so a new call started from `(te, ye)` does not detect the same crossing again.
- On a grid, a terminal stop returns `[t0, nodes <= te, te]`, without a repeated row when `te` is a node.

### `ContinuousCallback`

```python
ContinuousCallback(condition, affect=None, *, direction=0, terminal=False, record=False,
                   rootfind='left', save_positions=(True, True), interp_points=10,
                   repeat_nudge=0.01)
```

`condition(t, y, integ)` returns a float or a 1-D array of a fixed length, one value per component. `direction` and `terminal` are a scalar or an array with one entry per component. `affect(integ, idx)` receives the int array of the components that cross at the event time, and may change the state and the model as {ref}`changing the model <integrator_modification>` describes. `record=True` logs every crossing into `te`, `ye` and `ie`; at most one callback of a run records, and `opt.event` counts as one. `rootfind='left'` returns the last float before the crossing, suited to an `affect` that must act before it, and `'right'` the first float at which the component has crossed or is zero. `save_positions` saves the state before and after the event.

A ball that loses one tenth of its speed at each bounce is one callback and one call:

```python
from Solverz.integrator import solve, ContinuousCallback

def bounce(integ, idx):
    integ.u[1] = -0.9 * integ.u[1]

ground = ContinuousCallback(lambda t, y, integ: y[0], bounce, direction=-1)
sol = solve(nbball, [0, 30], y0, opt=Opt(rtol=1e-6, atol=1e-8), callbacks=[ground])
```

### How crossings are found

- A crossing is looked for after each accepted step, and the step is never shortened to find it. Each component is followed from the start of the step through `interp_points - 2` interior points of the step's interpolant, eight with the default `interp_points=10`, to the end of the step. A crossing that enters and leaves within one step is therefore found, unless it enters and leaves between two neighbouring points. Every crossing of a component that only records is found. Of a component that acts, the first crossing in its direction is found, also when it follows a crossing against its direction in the same step.
- The crossing is then located on the interpolant, to two adjacent floats, without a tolerance parameter. Its accuracy is that of the interpolant.
- **A component that is exactly zero at the start of a step does not cross there.** Nothing is therefore reported at the initial point of a call, and a run that starts exactly on a surface begins cleanly. A component that is zero at the start of a step and moves away from zero is not an event; its next crossing is, also when it lies in the same step.
- The earliest crossing of a component that acts on the run, because it is terminal or its callback has an `affect`, is one event instant `te` for all callbacks. Every component of any callback that crosses at `te` is handled there, and recorded crossings before `te` are logged in increasing time. After such an event, the next step is proposed with the length of the step that contained it.
- After an event found with `'left'`, a re-crossing by the same component within the first fraction `repeat_nudge` of the next step is the event already reported, not a new one.

### The cost of events

A continuous callback costs condition evaluations and interpolations on every accepted step. With `opt.event`, that is one call of `event` at the end of the step and one at each of eight interior points, nine calls against one for the legacy `Rodas`, and eight interpolations. Locating a crossing adds one call per iteration for each component that crosses at or before `te`. An event therefore costs about nine calls per step, however many components it has. `stats.ncondition` counts the calls. A smaller `interp_points` lowers the cost at the risk of missing a crossing that enters and leaves within one step; with `interp_points` below 3 only the two ends of the step are compared.

### `DiscreteCallback` and `preset_time_callback`

```python
DiscreteCallback(condition, affect, *, save_positions=(True, True), tstops=())
preset_time_callback(times, affect, *, save_positions=(True, True))
```

`condition(t, y, integ)` returns a bool and is evaluated after every accepted step, after the continuous callbacks; where it holds, `affect(integ)` runs. `tstops` join the stop times of the run. `preset_time_callback(times, affect)` runs `affect` at each of `times`, which are stop times, so that a step ends exactly on each. A scheduled change of the model, such as a fault applied at 0.1 s and cleared at 0.2 s, is exact in time and tolerance-independent this way, unlike a `TimeSeriesParam` profile that ramps between two values.

(integrator_modification)=

### Changing the model inside a run

The state or the model may change inside a call in two places only. An `affect` may change `integ.u`, entries of the arrays of `integ.dae.p` and `integ.dae.M.data`, and may call `integ.terminate()`. Between two `integ.step()` calls, the caller may change `integ.u`, an array of `dae.p` or `dae.M.data`, or rebind `dae.p` or `dae.M`, and then calls `integ.model_modified()`.

Both run the same modification protocol. It reads `dae.M` and `dae.p` again, resets the linear-solver cache to the state from which a new call starts, discards everything the method kept from earlier steps, makes the state consistent at the current time with `DaeIc`, and evaluates every condition again at the new state. A `DaeIc` failure ends the run as a failure. After `model_modified()`, `integ.interp` accepts only `tq == integ.t` until the next step, because the change may have overwritten the end of the step. A change made anywhere else, for example inside the residual, is not supported.

## The two configurations

Each Rosenbrock class runs in one of two configurations.

- The default configuration, `Rodas4()`, follows the rules of this page.
- The legacy-compatible configuration, `Rodas4(legacy_compat=True)`, and likewise for `Rodas3`, `Rodasp` and `Rodas5P`, reproduces the legacy `Rodas`: its step-size control, its error norm scaled by `|ynew|`, its `dF/dt`, its saving on a grid through the interpolant, and its end test. It refuses `tstops`, including those of a `DiscreteCallback`.

The legacy-compatible configuration returns `T` and `Y` byte-identical to the legacy `Rodas` when all of the following hold:

- no event and no callback;
- `opt.fix_h` is false, since the legacy fixed-step form can step past `tend`;
- `hinit` is `None` or positive;
- `t0 < tend`;
- the class matches `opt.scheme`, for example `Rodas3(legacy_compat=True)` for `scheme='rodas3'`;
- the legacy run neither raises nor stops on a factorization error;
- the `Opt` has not been passed to an earlier legacy `Rodas` call, which writes `hmax` and `facmax` into it.

With events and callbacks the legacy-compatible configuration follows the event rules of this page, not those of the legacy `Rodas`. The counters `nfeval`, `nJeval`, `ndecomp` and `nsolve` differ from the legacy ones in either configuration, since the core counts every call it makes.

(integrator_linsolver)=

## The linear solver

The backend, KLU or SuperLU, is chosen once per call, from `opt.linsolver` or else from the global selection of `set_linsolver`, `with linsolver(...)` or `SOLVERZ_LINSOLVER`. `init` resolves it when it is called, so a later `integ.solve()` outside a `with linsolver(...)` block keeps the backend of `init`. The iteration matrix is equilibrated by rows as in the legacy `Rodas`.

In the default configuration, a KLU symbolic analysis without row matching is shared between calls on one model, since it depends only on the sparsity pattern; the result of a call never depends on the calls made before it, on either backend. A dense Jacobian is factorized once per iteration matrix. The legacy-compatible configuration starts every call from an empty cache and solves a dense system at every stage, as the legacy `Rodas` does.

## Migrating from the legacy `Rodas`

The legacy `Rodas` is deprecated. It still works unchanged and warns with a `DeprecationWarning` at the line that calls it.

- `Rodas(dae, tspan, y0, opt)` becomes `Rodas4()(dae, tspan, y0, opt)`, or the class of `opt.scheme`: `Rodas3()`, `Rodasp()` or `Rodas5P()`, or `Rosenbrock.from_scheme(opt.scheme)`.
- To reproduce a legacy trajectory exactly, use `Rodas4(legacy_compat=True)` under the conditions of the previous section.
- The default configuration takes other steps than the legacy `Rodas`, so its trajectories agree with the legacy ones to the tolerance, not bit for bit. It scales the error by `max(|u|, |uprev|)`, scales the increment of the finite difference `dF/dt` with the step, lands exactly on `tend`, and shares the KLU analysis between calls.
- Differences in either configuration: a `DaeIc` failure ends the run with `ret == 'failed'` instead of raising `ValueError('Need Better y0')`; `hinit <= 0` and `fix_h` without `hinit` raise `ValueError` before any step; a factorization failure halves the step instead of stopping silently; `ret` is `'success'` or `'terminated'` on success instead of `None`; `ie` is int64; `T` and `Y` are new arrays, not views of a larger buffer; the output has no limit of 10001 rows; `Opt` is never written; a non-terminal event no longer shortens the step or adds a row; and a terminal event stops at the earliest terminal crossing, not at the first terminal component in index order.
- Differences of the default configuration only: a grid that does not increase strictly raises `ValueError` instead of ending the saving at its first repeated node, and the last step ends exactly on `tend`.

To silence the warning while code is migrated:

```python
import warnings
warnings.filterwarnings('ignore', 'Rodas is deprecated', DeprecationWarning)
```
