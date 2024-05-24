import numpy as np
import pandas as pd

from Solverz import DAE, Eqn, Ode, iVar, Para, idx, sin, cos, Rodas, as_Vars, Mat_Mul, \
    implicit_trapezoid, Opt, TimeSeriesParam, made_numerical

omega = iVar(name='omega')
delta = iVar(name='delta')
Ux = iVar(name='Ux')
Uy = iVar(name='Uy')
Ixg = iVar(name='Ixg')
Iyg = iVar(name='Iyg')
g = idx('g', value=[0, 1, 2])
ng = idx('ng', value=[3, 4, 5, 6, 7, 8])
Pm = Para(name='Pm')
G = Para(name='G', dim=2)
B = Para(name='B', dim=2)
D = Para(name='D')
Tj = Para(name='Tj')
ra = Para(name='ra')
omega_b = Para(name='omega_b')
Edp = Para(name='Edp')
Eqp = Para(name='Eqp')
Xdp = Para(name='Xdp')
Xqp = Para(name='Xqp')

rotator_eqn = Ode(name='rotator speed',
                  f=(Pm - (Ux[g] * Ixg + Uy[g] * Iyg + (Ixg ** 2 + Iyg ** 2) * ra) - D * (omega - 1)) / Tj,
                  diff_var=omega)
delta_eqn = Ode(name='delta', f=(omega - 1) * omega_b, diff_var=delta)
Ed_prime = Eqn(name='Ed_prime', eqn=Edp - sin(delta) * (Ux[g] + ra * Ixg - Xqp * Iyg) + cos(delta) * (
        Uy[g] + ra * Iyg + Xqp * Ixg))
Eq_prime = Eqn(name='Eq_prime', eqn=Eqp - cos(delta) * (Ux[g] + ra * Ixg - Xdp * Iyg) - sin(delta) * (
        Uy[g] + ra * Iyg + Xdp * Ixg))
Ix_inject = Eqn(name='Ixg_inject', eqn=Ixg - (Mat_Mul(G[g, :], Ux) - Mat_Mul(B[g, :], Uy)))
Iy_inject = Eqn(name='Iyg_inject', eqn=Iyg - (Mat_Mul(G[g, :], Uy) + Mat_Mul(B[g, :], Ux)))
Ixng_inject = Eqn(name='Ixng_inject', eqn=Mat_Mul(G[ng, :], Ux) - Mat_Mul(B[ng, :], Uy))
Iyng_inject = Eqn(name='Iyng_inject', eqn=Mat_Mul(G[ng, :], Uy) + Mat_Mul(B[ng, :], Ux))

dt = np.array(0.002)
T = np.array(10)

m3b9 = DAE([delta_eqn, rotator_eqn, Ed_prime, Eq_prime, Ix_inject, Iy_inject, Ixng_inject, Iyng_inject],
           name='m3b9')
m3b9.update_param('Pm', [0.7164, 1.6300, 0.8500])
m3b9.update_param('ra', [0.0000, 0.0000, 0.0000])
m3b9.update_param('D', [10, 10, 10])
m3b9.update_param('Tj', [47.2800, 12.8000, 6.0200])
m3b9.update_param('omega_b', [376.991118430775])
m3b9.update_param('g', [0, 1, 2])
m3b9.update_param('ng', [3, 4, 5, 6, 7, 8])
m3b9.update_param('Edp', [0.0000, 0.0000, 0.0000])
m3b9.update_param('Eqp', [1.05636632091501, 0.788156757672709, 0.767859471854610])
m3b9.update_param('Xdp', [0.0608, 0.1198, 0.1813])
m3b9.update_param('Xqp', [0.0969, 0.8645, 1.2578])
df = pd.read_excel('instances/test_m3b9.xlsx',
                   sheet_name=None,
                   engine='openpyxl',
                   header=None
                   )
Gvalue = np.asarray(df['G'])
m3b9.param_initializer('G', TimeSeriesParam(name='G',
                                            v_series=[Gvalue[6, 6], 10000, 10000, Gvalue[6, 6], Gvalue[6, 6]],
                                            time_series=[0, 0.002, 0.03, 0.032, T],
                                            value=Gvalue,
                                            index=(6, 6),
                                            dim=2))
m3b9.update_param('B', np.asarray(df['B']))

delta = iVar('delta', [0.0625815077879868, 1.06638275203221, 0.944865048677501])
omega = iVar('omega', [1, 1, 1])
Ixg = iVar('Ixg', [0.688836021737262, 1.57988988391346, 0.817891311823357])
Iyg = iVar('Iyg', [-0.260077644814056, 0.192406178191528, 0.173047791590276])
Ux = iVar('Ux',
          [1.04000110267534, 1.01157932564567, 1.02160343921907,
           1.02502063033405, 0.993215117729926, 1.01056073782038,
           1.02360471178264, 1.01579907336413, 1.03174403980626])
Uy = iVar('Uy',
          [9.38510394478286e-07, 0.165293826097057, 0.0833635520284917,
           -0.0396760163416718, -0.0692587531054159, -0.0651191654677445,
           0.0665507083524658, 0.0129050646926083, 0.0354351211556429])

y0 = as_Vars([delta, omega, Ixg, Iyg, Ux, Uy])

m3b9_dae, code = made_numerical(m3b9, y0, sparse=False, output_code=True)
sol_rodas = Rodas(m3b9_dae,
                  np.linspace(0, 10, 5001).reshape(-1, ),
                  y0,
                  Opt(hinit=1e-5))

sol_trape = implicit_trapezoid(m3b9_dae,
                               [0, 10],
                               y0,
                               Opt(step_size=dt))


def test_m3b9():
    assert np.abs(np.asarray(df['omega']) - sol_rodas.Y['omega']).max() <= 5e-5
    assert np.abs(np.asarray(df['omega']) - sol_trape.Y['omega']).max() <= 2.48e-4
    assert np.abs(np.asarray(df['delta']) - sol_rodas.Y['delta']).max() <= 9.6e-4
    assert np.abs(np.asarray(df['delta']) - sol_trape.Y['delta']).max() <= 7.8e-2
