# Example: Bouncing Ball (Minimal DAE with Events)

**What this teaches**: The simplest end-to-end Solverz workflow. An apple is launched up at 20 m/s, falls under gravity, and the simulation stops when it hits the ground.

**Equations**:
$$
\dot{v} = -9.8, \qquad \dot{h} = v
$$
with $v(0) = 20$, $h(0) = 0$. Stop on $h = 0$ (descending).

**Why this is canonical**: Demonstrates `Ode` declaration, `module_printer` rendering, event handling, and the basic solver-result API in ~30 lines. Exactly the form of the README's quick-start example.

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
from Solverz import Model, Var, Ode, Opt, made_numerical, Rodas

# 1. Symbolic model
m = Model()
m.h = Var('h', 0)
m.v = Var('v', 20)
m.f1 = Ode('f1', f=m.v, diff_var=m.h)      # dh/dt = v
m.f2 = Ode('f2', f=-9.8, diff_var=m.v)     # dv/dt = -9.8

# 2. Compile to numerical
bball, y0 = m.create_instance()              # type(bball) → DAE (because of Ode)
nbball = made_numerical(bball, y0, sparse=True)

# 3. Event: stop when h=0 going down
def events(t, y):
    value      = np.array([y[0]])    # y[0] is h (first declared variable)
    isterminal = np.array([1])       # stop on this event
    direction  = np.array([-1])      # only on descending zero crossing
    return value, isterminal, direction

# 4. Solve
sol = Rodas(nbball,
            np.linspace(0, 30, 100),
            y0,
            Opt(event=events))

# 5. Plot
plt.plot(sol.T, sol.Y['h'][:, 0])
plt.xlabel('Time / s')
plt.ylabel('h / m')
plt.show()

print(f"Hit ground at t = {sol.te[-1]:.3f} s, h = {sol.ye[-1][0]:.3e} m")
```

## Notes

- **Variable order matters for events**: `y[0]` is whichever `Var` was declared first (`h` in this case). `sol.Y['h']` is safer than indexing — it uses the variable name.
- **Event signature**: `def events(t, y)` returning `(value, isterminal, direction)` tuples of numpy arrays. Multiple events go in the array. `direction = -1` means "trigger only when crossing from positive to negative".
- **`sol.T` vs `sol.Y`**: `sol.T` is the time vector (1D). `sol.Y` is a `Vars`-like object — index by variable name to get the trajectory `(n_t, n_var)`.
- **`sol.te`, `sol.ye`, `sol.ie`**: event times, states at events, event indices.
- **Why `Rodas` and not `ode15s`**: Both work for this trivial system. `Rodas` is the default DAE solver in Solverz and supports event detection. `ode15s` is the BDF multistep alternative — better for very stiff problems.

## Variation: render to a module instead of inline

If you'd run this many times (e.g. parameter sweep), use `module_printer` instead of `made_numerical`:

```python
from Solverz import module_printer
printer = module_printer(bball, y0, name='bounceball', jit=True)
printer.render()

# Now in any file:
from bounceball import mdl as nbball, y as y0
sol = Rodas(nbball, np.linspace(0, 30, 100), y0, Opt(event=events))
```

The first call pays the Numba compile cost (cached on disk under `__pycache__/`). Subsequent runs skip compile entirely.
