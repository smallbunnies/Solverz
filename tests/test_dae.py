import pandas as pd
import numpy as np

from Solverz import Eqn, Ode, DAE, Rodas, implicit_trapezoid, Opt, made_numerical, Model, Var

m = Model()

m.x = Var('x', 1)
m.y = Var('y', 1)

m.f = Ode(name='f', f=-m.x ** 3 + 0.5 * m.y ** 2, diff_var=m.x)
m.g = Eqn(name='g', eqn=m.x ** 2 + m.y ** 2 - 2)
dae, y0 = m.create_instance()

ndae = made_numerical(dae, y0)

df = pd.read_excel('tests/dae_test.xlsx',
                   sheet_name=None,
                   engine='openpyxl'
                   )

sol_rodas = Rodas(ndae,
                  [0, 20],
                  y0,
                  opt=Opt(hinit=0.1))

sol_rodas_dense = Rodas(ndae,
                        np.linspace(0, 20, 201),
                        y0,
                        Opt(hinit=0.1))
sol_trape = implicit_trapezoid(ndae, [0, 20], y0, Opt(step_size=0.1))

sol_bench = Rodas(ndae,
                  np.linspace(0, 20, 201),
                  y0,
                  Opt(rtol=1e-6,
                      atol=1e-8))


def test_dae():
    xy_bench = np.asarray(df['rodas'])
    assert np.max(np.abs(xy_bench - sol_rodas.Y)) < 1e-8
    xy_bench1 = np.asarray(df['rodas_dense'])
    assert np.max(np.abs(xy_bench1 - sol_rodas_dense.Y)) < 1e-8
    assert np.max(np.abs(sol_bench.Y.array - sol_trape.Y.array)) < 7e-4
