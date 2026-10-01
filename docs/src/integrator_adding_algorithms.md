(integrator_adding_algorithms)=

# Adding an algorithm to the integrator

The {ref}`integrator core <integrator>` owns step-size control, acceptance and rejection, the output grid, stop times, events, the linear algebra, the counters and the failure path. An algorithm states only how one step is taken, as the method `perform_step` of one class in one file. Nothing is registered and nothing in the core changes: the instance is passed as `solve(dae, tspan, y0, alg=MyMethod())`, or called as `MyMethod()(dae, tspan, y0, opt)`.

A method is written in one of three styles.

- **Table.** A Rosenbrock method is its coefficient table. In this release the Rosenbrock family is the only family defined by tables; explicit Runge-Kutta and SDIRK methods are written as formulas.
- **Formula.** `perform_step(self, s)` computes the step out of place with the services of `s` and returns the new state, and its error estimate when it has one. `ImplicitEuler` and `Trapezoid` are written this way.
- **In place.** `perform_step(self, integ, cache)` writes into buffers allocated once, as the built-in Rosenbrock methods do. It is the fastest style and the hardest to read.

## The contract

An algorithm is a subclass of `Solverz.integrator.Algorithm`.

| Member | Default | Meaning |
|---|---|---|
| `scheme` | required | a `str` that names the method in `Stats` and in messages |
| `order` | required | the order of the method, a positive int |
| `error_order` | `order` | the error estimate is `O(h**error_order)`; the controller exponent is `1/error_order` |
| `interp_order` | 1 | the order of the interpolant, which `check_algorithm` measures |
| `adaptive` | `False` | `True` when `perform_step` returns an error estimate |
| `explicit` | `False` | `True` when the method never solves with the iteration matrix and uses `s.f` instead |
| `inplace` | `False` | the style of `perform_step` |
| `norm` | `'rms'` | the error norm, `'rms'` or `'max'` |
| `perform_step(self, s)` | required | formula style: return `y` or `(y, err)` |
| `perform_step(self, integ, cache)` | | in-place style, with `inplace = True`: write `integ.u` and, in an adaptive run, `integ.EEst` |
| `alloc(self, integ)` | an empty namespace | allocate the buffers of the method once per call; the result is `s.cache` |
| `interpolant(self, integ, cache, theta, out)` | linear | the state at `tprev + theta * dt_step`, written into `out` or returned |
| `addsteps(self, integ, cache)` | nothing | compute once per step what the interpolant needs beyond the step itself |
| `reset_history(self, integ, cache)` | nothing | clear every datum kept from earlier steps; the core calls it whenever the state or the model changed |
| `controller(self, opts)` | `IController(opts, self)` | the step-size controller of a run |
| `initial_dt(self, integ)` | `hinit`, else `1e-6 * (tend - t0)` | the first step |

`Integrator` raises `TypeError` before the first step when `perform_step` is missing or does not take one parameter after `self` with `inplace = False` or two with `inplace = True`. In the formula style it raises `TypeError` when `y` or `err` is not a vector of the length of the state, when an adaptive method returns no error estimate, and when a method that declares `adaptive = False` returns one without `opt.fix_h`. These are programming errors and are reported at once.

A method without an error estimate declares `adaptive = False`. It then runs with the fixed step `opt.hinit`, which is required, and lands exactly on the stop times and on `tend`. A failed step of such a run fails the run, since the step cannot shrink.

The hooks `interpolant`, `addsteps` and `reset_history` take `(integ, cache)` in both styles. They describe the last accepted step through `integ.tprev`, `integ.uprev`, `integ.t_step`, `integ.u_step` and `integ.dt_step`, and never read `integ.u`, which an event may have replaced. `integ.ctx` gives them the services of the formula style. The default interpolant is linear between `uprev` and `u_step`, which is safe for algebraic variables; a better interpolant raises the accuracy of the output grid and of event location to its order.

## `ImplicitEuler`, in full

This is the whole file `Solverz/integrator/implicit_euler.py`:

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

`s.implicit(t, gamma, rhs)` returns the `y` with `M y - h*gamma*F(t, y) = rhs`, which for `gamma = 1` and `rhs = M y0` is the backward Euler step. The error estimate reuses the factorization that the Newton iteration has already computed, so it costs one solve. On the algebraic rows the right-hand side is zero, since `M` has zero rows there and `D` removes the algebraic rows of `F0`, so the Newton iteration enforces `F(t0 + h, y1) = 0` on those rows and the estimate is zero there. Without `D`, the small algebraic residual that `DaeIc` accepts at the start would enter the estimate at every step size, and a run with a small `atol` would fail with too many rejections.

The trapezoidal rule is as short:

```python
class Trapezoid(Algorithm):
    scheme = 'trapezoid'
    order = 2
    error_order = 2
    adaptive = True

    def perform_step(self, s):
        f0 = s.D * s.F0
        y = s.implicit(s.t + s.h, 0.5, s.M @ s.y0 + 0.5 * s.h * f0)
        return y, s.W(0.5).solve(s.M @ (y - s.y0) - s.h * f0)
```

## The services

`s` is the `StepContext` of the Integrator. Its members are named after the symbols of the formulas.

| Member | Meaning |
|---|---|
| `s.t`, `s.h` | the start of the attempt and its step |
| `s.new_step` | `True` on the first attempt of a step, `False` on its retries |
| `s.y0` | a read-only view of the state at the start of the step; valid during the attempt only, so an algorithm that keeps it copies it |
| `s.n`, `s.M`, `s.p` | the size, the mass matrix and the parameters; `M` is never modified |
| `s.D` | 1.0 on the rows of `M` that hold a nonzero value and 0.0 on the algebraic rows, so `s.D * v` keeps the differential rows of `v` |
| `s.rtol`, `s.atol`, `s.adaptive` | the tolerances of the run, and whether it is adaptive |
| `s.cache` | the result of `alloc` |
| `s.F(t, y, out=None)` | the residual |
| `s.F0` | a read-only view of `F(t, y0)`, evaluated once per step |
| `s.f(t, y, out=None)` | the derivative `M^-1 F(t, y)`, for a model whose `M` pairs every row with exactly one variable |
| `s.J(t, y)` | the Jacobian, evaluated at every call |
| `s.J0` | the Jacobian at `(t, y0)`, evaluated once per step; the matrix of `s.W` |
| `s.dFdt(out=None)` | a read-only view of the partial derivative of `F` with respect to `t` at `(t, y0)`, computed once per step |
| `s.W(gamma)` | the factorization of `M - (h*gamma) J0`, kept for the attempt; `.solve(b, out=None)` solves with it |
| `s.implicit(t, gamma, rhs, y=None, out=None, slope=False)` | the `y` with `M y - h*gamma*F(t, y) = rhs`, from `y` or `y0`; with `slope=True`, `(y, k)` with `k = (M y - rhs) / (h*gamma)`, which equals `F(t, y)` to the Newton tolerance |
| `s.error_norm(e)` | the scalar error of the vector `e` that the controller reads |

The services count every evaluation, factorization and solve in `Stats`; an algorithm never touches `Stats`. `out=` is accepted wherever a vector is returned and is never required; with it, `F` and `f` allocate nothing. The same services exist on the Integrator as `integ.F`, `integ.f`, `integ.D()`, `integ.F0()`, `integ.J`, `integ.J0()`, `integ.dFdt`, `integ.W`, `integ.implicit` and `integ.error_norm`, for in-place algorithms.

`s.implicit` runs a simplified Newton iteration with the matrix `s.W(gamma)`. It stops when the estimated Newton error is one percent of the tolerance, in the tolerance-weighted root mean square norm, and raises `StepFailure` when the iteration diverges, converges too slowly or produces a value that is not finite. The factorization is computed once per `gamma` per attempt, so several `implicit` calls with one `gamma` share it; an SDIRK method is one `s.implicit(..., slope=True)` call per stage.

An explicit method declares `explicit = True` and uses `s.f` instead of `s.W`. It can integrate only a model whose mass matrix pairs every row with exactly one variable, an ODE whose equations may be declared in any order; on any other model the Integrator raises `TypeError` before the first step, so that an explicit formula never moves a variable by the residual of another row.

### Failed attempts and retries

An algorithm signals a failed attempt by raising `StepFailure`; the services raise it on a failed factorization or solve and on a failed Newton iteration, and an arithmetic error of the model is turned into it. The core rejects the attempt and retries with half the step. On a retry of the same step, `s.t` and `s.y0` are unchanged, `s.new_step` is `False`, `s.h` is the step the controller set, and `s.F0`, `s.J0` and `s.dFdt()` return the values of the first attempt. `s.cache` holds whatever the rejected attempt wrote, possibly only in part.

### The history rule

When `s.new_step` is true, `s.cache` holds exactly what the last accepted attempt left in it, unless `reset_history` ran after it. The core keeps no data of earlier steps for an algorithm. An algorithm that uses data of the previous accepted step, such as the end slope of a first-same-as-last method or a Newton starting value, keeps it in `s.cache`, copies what it keeps, and clears all of it in `reset_history`. The core calls `reset_history` whenever an `affect` or `model_modified()` changes the state or the model, and the history check of `check_algorithm` detects data that survive such a change.

## A Rosenbrock method by its table

A Rosenbrock method needs only its table. `RosenbrockTableau.from_hairer` takes the table in Hairer's notation, with either `beta`, the table of `alpha_ij + gamma_ij`, or `gamma_ij`, the coupling coefficients as published for ROS methods:

```python
from Solverz.integrator import Rosenbrock, RosenbrockTableau

class MyRos(Rosenbrock):
    scheme = 'myros'
    interp_order = 1
    tableau = RosenbrockTableau.from_hairer(
        gamma=..., alpha=[[0, 0, 0], [..., 0, 0], [..., ..., 0]],
        beta=[[0, 0, 0], [..., 0, 0], [..., ..., 0]],
        b=[...], bd=[...], pord=3)
```

`perform_step`, the controller, the norm and the interpolant come from `Rosenbrock`. The dense output of the table is used when `c`, `d` and `e` are given, and the linear interpolant otherwise. `order` and `error_order` are the table's `pord`.

## The controller

`Algorithm.controller` returns the integral controller `IController`. An algorithm that prefers the PI controller of OrdinaryDiffEq.jl writes

```python
def controller(self, opts):
    return PIController(opts, self)
```

A controller keeps its state in its own object, which belongs to one call, and reads its parameters from the options of the call; see `Controller` in the {ref}`API reference <reference>`.

## Checking an algorithm

```python
from Solverz.integrator.testing import check_algorithm

check_algorithm(MyMethod())
```

`check_algorithm(alg, *, order_tol=0.3, rendered=False)` runs the conformance checks on the instance `alg` and raises `AssertionError('check_algorithm[<check>]: ...')` at the first that fails. It returns the values each check measured. The checks build their own small models and depend only on the declared traits: an explicit method runs on an ODE, and a method without an error estimate runs with the fixed step `2**-6`.

| Check | What it asserts |
|---|---|
| contract | `perform_step` has the signature of `inplace`; `order`, `error_order` and `interp_order` are positive; an explicit method refuses a DAE |
| order | on fixed-step runs with `h = 2**-1` to `2**-6`, the order measured from the largest error over every variable, the algebraic ones included, reaches `order - order_tol`, on a DAE whose rows of `M` are aligned with the variables and on one whose rows are not |
| interpolant | the interpolant returns the start of the step exactly and its end within `1e-10` relative, and its order, measured from the largest error over every variable inside the steps, reaches `interp_order - order_tol` |
| saveat | neither an output grid nor interpolation between steps changes the steps, and every saved node is the interpolant's value there |
| tstops | stop times are landed on exactly |
| events | a terminal event with two identical components, on a ball that starts on their root, stops at one time near the impact, records both, reports nothing at `t0`, and returns its state as the last row |
| events on a grid | the rows before a terminal event are those of the same run without the event |
| algebraic event | a terminal event where the algebraic variable `z = sin t` reaches 0.5 stops the run near `pi/6`, located on the interpolant of `z`; an explicit method skips it |
| inconsistent start | a start with a small algebraic residual that `DaeIc` leaves in place succeeds at `atol = 1e-10`; an explicit method skips it |
| history | a run changed at 0.5 by a callback equals, after 0.5, two calls split there |
| failure | a solution that blows up ends the run as failed with one printed line and no exception |
| out | a residual without `out` gives the result of one with it |
| Opt | ten calls with one `Opt` leave it unchanged and give one result |
| y0 | the initial state is never written, as an ndarray and as a `Vars` |
| counts | the counters equal the calls of `F` and `J` |
| rendered | with `rendered=True`, a model rendered by `module_printer` agrees with the inline one |

The last check compiles the rendered model with Numba and is therefore opt-in. A method that passes every check integrates DAEs, lands on stop times, locates events on its interpolant, survives changes of the model and fails cleanly, with no code in the core written for it.
