# Example: Gas Pipeline by Method of Characteristics (Canonical FDAE)

**What this teaches**: Discretizing a 1D isothermal gas pipeline PDE with the **method of characteristics** (MOC) and solving the resulting time-stepping problem with `fdae_solver`. Demonstrates `AliasVar` (previous-time-step state) and `TimeSeriesParam` (boundary condition).

**Why this is canonical**: It's the smallest end-to-end FDAE example. Any time you have a hyperbolic PDE (gas flow, water hammer, traffic flow, etc.) and want a fixed-step explicit time stepping scheme that uses last-step state as the right-hand side, this is the pattern.

**Solver**: `fdae_solver` with `Opt(step_size=dt)` to enforce the MOC's CFL-required step size.

## Math

Isothermal gas pipeline equations:

$$
\frac{\partial p}{\partial t} + \frac{a^2}{S} \frac{\partial q}{\partial x} = 0,
\qquad
\frac{\partial q}{\partial t} + S \frac{\partial p}{\partial x} + \frac{\lambda a^2 q |q|}{2 D S p} = 0
$$

where $p$ is pressure, $q$ is mass flow, $a$ is the speed of sound, $S$ is the pipe cross-section, $D$ is diameter, and $\lambda$ is the Darcy friction factor.

Method of characteristics: along $\mathrm{d}x/\mathrm{d}t = \pm a$ the PDE collapses to two ODEs that can be discretized as algebraic equations relating values at neighboring grid nodes between two time steps. The CFL constraint forces `dt = dx / a`.

## Code

```python
import numpy as np
from sympy import Integer

from Solverz import (Var, Param, Eqn, Opt, Abs,
                     made_numerical, TimeSeriesParam, Model,
                     AliasVar, fdae_solver)

# === 1. Physical parameters ===
L  = 51000 * 0.8                  # pipe length (m)
p0 = 6621246.69079594             # initial pressure (Pa)
q0 = 14                           # initial mass flow (kg/s)
va = Integer(340)                 # speed of sound (m/s)
D  = 0.5901                       # pipe diameter (m)
S  = np.pi * (D / 2) ** 2         # cross section (m²)
lam = 0.03                        # Darcy friction factor

dx = 500
dt = 1.4706                       # CFL: dt = dx / va
M  = int(L / dx)                  # number of pipe segments

# === 2. Build the FDAE model ===
m1 = Model()

# Current and previous-step states
m1.p  = Var('p',  value=p0 * np.ones((M + 1,)))
m1.q  = Var('q',  value=q0 * np.ones((M + 1,)))
m1.p0 = AliasVar('p', init=m1.p)   # previous time step value of p
m1.q0 = AliasVar('q', init=m1.q)   # previous time step value of q

# === 3. Characteristic equations ===
# Forward characteristic (downstream-going wave) at nodes 1..M
m1.ae1 = Eqn(
    'cha1',
    m1.p[1:M + 1] - m1.p0[0:M]
    + va / S * (m1.q[1:M + 1] - m1.q0[0:M])
    + lam * va**2 * dx / (4 * D * S**2)
      * (m1.q[1:M + 1] + m1.q0[0:M]) * Abs(m1.q[1:M + 1] + m1.q0[0:M])
      / (m1.p[1:M + 1] + m1.p0[0:M])
)

# Backward characteristic (upstream-going wave) at nodes 0..M-1
m1.ae2 = Eqn(
    'cha2',
    m1.p0[1:M + 1] - m1.p[0:M]
    + va / S * (m1.q[0:M] - m1.q0[1:M + 1])
    + lam * va**2 * dx / (4 * D * S**2)
      * (m1.q[0:M] + m1.q0[1:M + 1]) * Abs(m1.q[0:M] + m1.q0[1:M + 1])
      / (m1.p[0:M] + m1.p0[1:M + 1])
)

# === 4. Boundary conditions ===
# Left BC: pressure profile (drops sharply at t = 1000 s)
T = 5 * 3600
pb1 = 1e6
pb_t   = [p0,    p0,        pb1,             pb1]
tseries = [0,   1000, 1000 + 10*dt,           T]
m1.pb = TimeSeriesParam('pb',
                        v_series=pb_t,
                        time_series=tseries)

# Right BC: constant outflow
m1.qb = Param('qb', q0)

m1.bd1 = Eqn('bd1', m1.p[0]  - m1.pb)   # left:  p[0] = pb(t)
m1.bd2 = Eqn('bd2', m1.q[M]  - m1.qb)   # right: q[M] = qb

# === 5. Compile + solve ===
fdae, y0 = m1.create_instance()                # type: FDAE
nfdae = made_numerical(fdae, y0, sparse=True)

sol = fdae_solver(nfdae, [0, T], y0, Opt(step_size=dt))

# === 6. Extract ===
import matplotlib.pyplot as plt
# Plot inlet and outlet pressure over time
plt.plot(sol.T, sol.Y['p'][:, 0],   label='p[0] (inlet)')
plt.plot(sol.T, sol.Y['p'][:, -1],  label='p[M] (outlet)')
plt.xlabel('Time / s'); plt.ylabel('Pressure / Pa')
plt.legend(); plt.show()
```

## Notes

- **`AliasVar('p', init=m1.p)`** — declares `m1.p0` as the historical value of `m1.p` from the previous step. The `name='p'` (matching the live variable) tells Solverz they're paired. The `init=m1.p` sets the initial history to the live initial condition. **Solverz auto-detects this declaration and the resulting model is an `FDAE`** (not an `AE`).
- **No `Ode`** — FDAE does not use `Ode`. Time stepping is encoded in the structure of the equations themselves: `m.q[i] - m.q0[i] + ...` says "the new value relates to the old value via this discrete formula."
- **Slicing** — `m.p[1:M+1] - m.p0[0:M]` produces an `M`-element vector equation. The slice indices match the MOC stencil (forward characteristic uses upstream history at `i` and downstream current at `i+1`).
- **CFL: `dt = dx / va`** — required for MOC stability. Pass to the solver via `Opt(step_size=dt)`. The solver does NOT compute its own step size for FDAE.
- **`TimeSeriesParam` for boundary conditions** — for a step change, use a steep but finite ramp (`1000` → `1000 + 10*dt`). A truly vertical jump can cause the solver to bracket the discontinuity awkwardly.
- **Why not `Rodas`** — `Rodas` integrates `M y' = F(t, y, p)` with adaptive step. `fdae_solver` integrates `0 = F(t, y, y_prev, p)` with a fixed step. Hyperbolic PDEs discretized by MOC are naturally FDAEs, not DAEs.

## See also

- Cookbook chapter: `fdae/cha/cha.md`
- For implicit time stepping of gas / heat dynamics, see `SolMuseum.pde.gas` and `SolMuseum.pde.heat` for prebuilt discretization helpers.
- For coupled gas + power dynamics in a full IES, see `examples/m3b9-dynamics.md` (DAE) and the cookbook `dae/ies/ies.md`.
