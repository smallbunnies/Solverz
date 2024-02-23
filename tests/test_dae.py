import pandas as pd
import numpy as np

from Solverz import Eqn, Ode, DAE, Var, as_Vars, Rodas, implicit_trapezoid, Opt, made_numerical

x = Var('x', 1)
y = Var('y', 1)

f = Ode(name='f', f=-x ** 3 + 0.5 * y ** 2, diff_var=x)
g = Eqn(name='g', eqn=x ** 2 + y ** 2 - 2)
dae = DAE([f, g])

z = as_Vars([x, y])
ndae = made_numerical(dae, z)

df = pd.read_excel('instances/dae_test.xlsx',
                   sheet_name=None,
                   engine='openpyxl'
                   )

T, Z, stats = Rodas(ndae,
                    [0, 20],
                    z,
                    opt=Opt(hinit=0.1))

T1, Z1, stats1 = Rodas(ndae,
                       np.linspace(0, 20, 201),
                       z,
                       Opt(hinit=0.1))
T2, Z2, stats2 = implicit_trapezoid(ndae, [0, 20], z, Opt(step_size=0.1))

T3, Zbench, stats3 = Rodas(ndae,
                           np.linspace(0, 20, 201),
                           z,
                           Opt(rtol=1e-6,
                               atol=1e-8))


def test_dae():
    xy_bench = np.asarray(df['rodas'])
    assert np.max(np.abs(xy_bench - Z.array)) < 1e-8
    xy_bench1 = np.asarray(df['rodas_dense'])
    assert np.max(np.abs(xy_bench1 - Z1.array)) < 1e-8
    assert np.max(np.abs(Zbench - Z2.array)) < 7e-4
