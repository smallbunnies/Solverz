# Example: M3B9 Power System Dynamics (Canonical DAE with Events)

**What this teaches**: A 3-machine, 9-bus power system electromechanical transient simulation. Combines `Ode` (rotor speed and angle dynamics) with `Eqn` (algebraic generator internal voltages + network injections). Demonstrates `TimeSeriesParam` for fault scenarios — bus 6 self-conductance jumps to 10000 between $t = 0.002$ s and $t = 0.03$ s to model a three-phase fault.

**Why this is canonical**: It's the standard "Anderson-Fouad" benchmark for power system stability, the smallest realistic DAE that combines:
- ODE rotor dynamics (`Ode(name, f, diff_var=...)`)
- Algebraic generator equations (`Eqn(...)`)
- Algebraic network injections (large `Eqn` block, one per bus)
- Time-varying conductance to model a fault (`TimeSeriesParam`)

**Solver**: `Rodas` (stiffly-accurate Rosenbrock with adaptive step + dense output). `Opt(hinit=1e-5)` to clamp the initial step small enough to capture the fault.

## Code

```python
import numpy as np
from Solverz import (Eqn, Ode, Var, Param, sin, cos, Rodas, Opt,
                     TimeSeriesParam, made_numerical, Model)

# === 1. Load network admittance (G, B = real/imag of Ybus) ===
# G, B are 9x9 numpy arrays loaded from the case file
# (omitted here for brevity)

# === 2. Build the model ===
m = Model()

# --- State variables ---
m.omega = Var('omega', [1, 1, 1])                                   # rotor speed (3 machines)
m.delta = Var('delta', [0.0625815077879868,
                        1.06638275203221,
                        0.944865048677501])                          # rotor angle (3 machines)
m.Ux = Var('Ux', [1.04000110267534, 1.01157932564567,
                  1.02160343921907, ...])                           # bus voltage real part (9 buses)
m.Uy = Var('Uy', [9.38510394478286e-07, 0.165293826097057,
                  0.0833635520284917, ...])                          # bus voltage imag part (9 buses)
m.Ixg = Var('Ixg', [0.688836021737262,
                    1.57988988391346,
                    0.817891311823357])                              # generator current real (3)
m.Iyg = Var('Iyg', [-0.260077644814056,
                    0.192406178191528,
                    0.173047791590276])                              # generator current imag (3)

# --- Machine parameters ---
m.Pm  = Param('Pm',  [0.7164, 1.6300, 0.8500])
m.D   = Param('D',   [10, 10, 10])
m.Tj  = Param('Tj',  [47.2800, 12.8000, 6.0200])
m.ra  = Param('ra',  [0.0, 0.0, 0.0])
m.Edp = Param('Edp', [0.0, 0.0, 0.0])
m.Eqp = Param('Eqp', [1.05636632091501, 0.788156757672709, 0.767859471854610])
m.Xdp = Param('Xdp', [0.0608, 0.1198, 0.1813])
m.Xqp = Param('Xqp', [0.0969, 0.8645, 1.2578])
wb = 376.991118430775

# --- Rotor dynamics (Ode) ---
Pe = m.Ux[0:3] * m.Ixg + m.Uy[0:3] * m.Iyg + (m.Ixg ** 2 + m.Iyg ** 2) * m.ra
m.rotator_eqn = Ode(name='rotator speed',
                    f=(m.Pm - Pe - m.D * (m.omega - 1)) / m.Tj,
                    diff_var=m.omega)

omega_coi = (m.Tj[0]*m.omega[0] + m.Tj[1]*m.omega[1] + m.Tj[2]*m.omega[2]) \
            / (m.Tj[0] + m.Tj[1] + m.Tj[2])
m.delta_eq = Ode('delta equation',
                 wb * (m.omega - omega_coi),
                 diff_var=m.delta)

# --- Generator internal voltage equations (algebraic) ---
m.Ed_prime = Eqn('Ed_prime',
                 (m.Edp - sin(m.delta) * (m.Ux[0:3] + m.ra*m.Ixg - m.Xqp*m.Iyg)
                        + cos(m.delta) * (m.Uy[0:3] + m.ra*m.Iyg + m.Xqp*m.Ixg)))
m.Eq_prime = Eqn('Eq_prime',
                 (m.Eqp - cos(m.delta) * (m.Ux[0:3] + m.ra*m.Ixg - m.Xdp*m.Iyg)
                        - sin(m.delta) * (m.Uy[0:3] + m.ra*m.Iyg + m.Xdp*m.Ixg)))

# --- Fault: bus 6 self-conductance G[6,6] surges 0.002 ≤ t ≤ 0.03 s ---
m.G66 = TimeSeriesParam('G66',
                        v_series=[G[6, 6], 10000, 10000, G[6, 6], G[6, 6]],
                        time_series=[0, 0.002, 0.03, 0.032, 10])

def getGitem(r, c):
    if r == 6 and c == 6:
        return m.G66
    return G[r, c]

# --- Network injection equations (one per bus, rectangular form) ---
for i in range(9):
    rhs1 = m.Ixg[i] if i < 3 else 0
    for j in range(9):
        rhs1 = rhs1 - getGitem(i, j) * m.Ux[j] + B[i, j] * m.Uy[j]
    m.__dict__[f'Ix_inj_{i}'] = Eqn(f'Ix injection {i}', rhs1)

for i in range(9):
    rhs2 = m.Iyg[i] if i < 3 else 0
    for j in range(9):
        rhs2 = rhs2 - getGitem(i, j) * m.Uy[j] - B[i, j] * m.Ux[j]
    m.__dict__[f'Iy_inj_{i}'] = Eqn(f'Iy injection {i}', rhs2)

# === 3. Compile + solve ===
m3b9, y0 = m.create_instance()                                       # type: DAE
mdl = made_numerical(m3b9, y0, sparse=True)

sol = Rodas(mdl,
            np.linspace(0, 10, 1001),
            y0,
            Opt(hinit=1e-5))

# === 4. Extract trajectories ===
import matplotlib.pyplot as plt
plt.plot(sol.T, sol.Y['omega'])
plt.xlabel('Time / s'); plt.ylabel('Rotor speed (pu)')
plt.legend(['Gen 1', 'Gen 2', 'Gen 3'])
plt.show()
```

## Notes

- **Why `m.Ode` AND `m.Eqn`**: Solverz auto-detects the equation type. Models with at least one `Ode` become `DAE`. The `Ode`s become the differential rows (`M y' = ...`), the `Eqn`s become the algebraic rows.
- **`TimeSeriesParam` semantics**: linear interpolation between `(time_series, v_series)` knots. For a step function (like a fault), use a near-vertical jump: `time_series=[0, 0.002, 0.03, 0.032, 10]` makes the value jump in 0.0005 s windows on either side. The 0.002-second ramp is short enough that Rodas's adaptive stepper crosses it at the smallest possible step.
- **`Opt(hinit=1e-5)`**: prevents the solver from taking a huge first step that would skip over the fault entirely. Without it, Rodas's automatic initial-step heuristic might pick `h = 0.1 s`.
- **`sol.Y['omega']` vs `sol.y['omega']`**: DAE solvers return `sol.Y` (an object with one trajectory per variable, indexed by name). AE solvers return `sol.y` (lowercase, single solution).
- **For events** (e.g. relay trip when `omega > 1.05`): pass `Opt(event=event_fn)` — see `examples/bouncing-ball.md` for the event signature.
- **Initial conditions matter**: the `Var` initial values must satisfy all algebraic constraints at $t = 0$. They're typically computed from a separate steady-state power flow (use `SolUtil.PowerFlow`). Mismatched IC will produce a "Inconsistent initial values for algebraic equation" warning.

## Going larger

For a real integrated energy system (multiple generators, heat network, gas network, all coupled), use the SolMuseum prebuilt blocks instead of writing the equations by hand:

```python
from SolMuseum.dae import gt, pv, st, eb, heat_network, gas_network
from SolMuseum.ae import eps_network

model = Model()
model.add(gt(ux=..., uy=..., ix=..., iy=..., ...).mdl())
model.add(pv(ux=..., uy=..., ...).mdl())
# ... etc
model.add(eps_network(pf).mdl(dyn=True))
spf, y0 = model.create_instance()
```

See `references/ecosystem.md` for the full SolMuseum block list and the cookbook `dae/ies/ies.md` chapter for the end-to-end IES example.

## See also

- Cookbook chapter: `dae/m3b9/m3b9.md`
- IES (multi-component) example: `dae/ies/ies.md`
- SolMuseum blocks for power system dynamics: `references/ecosystem.md`
