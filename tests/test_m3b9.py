import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.algebra import AliasVar, ComputeParam, F, X, Y
from core.eqn import Ode, Eqn
from core.equations import DAE
from core.param import Param
from core.solver import nr_method, implicit_trapezoid
from core.var import TimeVar
from core.variables import TimeVars
from core.event import Event

rotator = Ode(name='rotator speed', e_str='(Pm-(Uxg*Ixg+Uyg*Iyg+(Ixg**2+Iyg**2)*ra)-D*(omega-1))/Tj', diff_var='omega')
delta = Ode(name='delta', e_str='(omega-1)*omega_b', diff_var='delta')
generator_ux = Eqn(name='Generator Ux', e_str='Uxg-Ag*Ux', commutative=False)
generator_uy = Eqn(name='Generator Uy', e_str='Uyg-Ag*Uy', commutative=False)
Ed_prime = Eqn(name='Ed_prime', e_str='Edp-sin(delta)*(Uxg+ra*Ixg-Xqp*Iyg)+cos(delta)*(Uyg+ra*Iyg+Xqp*Ixg)')
Eq_prime = Eqn(name='Eq_prime', e_str='Eqp-cos(delta)*(Uxg+ra*Ixg-Xdp*Iyg)-sin(delta)*(Uyg+ra*Iyg+Xdp*Ixg)')
Ixg_inject = Eqn(name='Ixg_inject', e_str='Ixg-(Mat_Mul(Ag*G,Ux)-Mat_Mul(Ag*B,Uy))', commutative=False)
Iyg_inject = Eqn(name='Iyg_inject', e_str='Iyg-(Mat_Mul(Ag*G,Uy)+Mat_Mul(Ag*B,Ux))', commutative=False)
Ixng_inject = Eqn(name='Ixng_inject', e_str='Mat_Mul(Ang*G,Ux)-Mat_Mul(Ang*B,Uy)', commutative=False)
Iyng_inject = Eqn(name='Iyng_inject', e_str='Mat_Mul(Ang*G,Uy)+Mat_Mul(Ang*B,Ux)', commutative=False)

param = [Param('Pm'), Param('ra'), Param('D'), Param('Tj'), Param('omega_b'), Param('Ag'), Param('Edp'), Param('Eqp'),
         Param('Xqp'), Param('Xdp'), Param('G'), Param('B'), Param('Ang')]

m3b9 = DAE([rotator, delta, generator_ux, generator_uy, Ed_prime, Eq_prime, Ixg_inject, Iyg_inject,
            Ixng_inject, Iyng_inject],
           name='m3b9',
           param=param)
m3b9.update_param('Pm', [0.7164, 1.6300, 0.8500])
m3b9.update_param('ra', [0.0000, 0.0000, 0.0000])
m3b9.update_param('D', [50, 50, 50])
m3b9.update_param('Tj', [47.2800, 12.8000, 6.0200])
m3b9.update_param('omega_b', [376.991118430775])
Ag = np.zeros((3, 9))
Ag[0, 0] = 1
Ag[1, 1] = 1
Ag[2, 2] = 1
m3b9.update_param('Ag', Ag)
Ang = np.zeros((6, 9))
Ang[np.ix_([0, 1, 2, 3, 4, 5], [3, 4, 5, 6, 7, 8])] = np.eye(6)
m3b9.update_param('Ang', Ang)
m3b9.update_param('Edp', [0.0000, 0.0000, 0.0000])
m3b9.update_param('Eqp', [1.05636632091501, 0.788156757672709, 0.767859471854610])
m3b9.update_param('Xdp', [0.0608, 0.1198, 0.1813])
m3b9.update_param('Xqp', [0.0969, 0.8645, 1.2578])
df = pd.read_excel('../instances/test_m3b9.xlsx',
                   sheet_name=None,
                   engine='openpyxl',
                   header=None
                   )
m3b9.update_param('G', np.asarray(df['G']))
m3b9.update_param('B', np.asarray(df['B']))
delta = TimeVar('delta')
delta.v = [0.0625815077879868, 1.06638275203221, 0.944865048677501]
omega = TimeVar('omega')
omega.v = [1, 1, 1]
Ixg = TimeVar('Ixg')
Ixg.v = [0.688836025963652, 1.57988988520965, 0.817891311270257]
Iyg = TimeVar('Iyg')
Iyg.v = [-0.260077649504215, 0.192406179350779, 0.173047793408992]
Ux = TimeVar('Ux')
Ux.v = [1.04000110485564, 1.01157932697927, 1.02160344047696, 1.02502063224420, 0.993215119385170, 1.01056073948763,
        1.02360471318869, 1.01579907470994, 1.03174404117073]
Uxg = TimeVar('Uxg')
Uxg.v = [1.04000110485564, 1.01157932697927, 1.02160344047696]
Uy = TimeVar('Uy')
Uy.v = [9.38266043832060e-07, 0.165293825576865, 0.0833635512998016, -0.0396760168295497, -0.0692587537381123,
        -0.0651191661122707, 0.0665507077512621, 0.0129050640060066, 0.0354351204593645]
Uyg = TimeVar('Uyg')
Uyg.v = [9.38266043832060e-07, 0.165293825576865, 0.0833635512998016]

dt = np.array(0.003)
T = np.array(3)

e1 = Event(name='three phase fault', time=[0, 0.03, T])
e1.add_var('G', [10000, np.asarray(df['G'])[6, 6], np.asarray(df['G'])[6, 6]], (6, 6))

xy = implicit_trapezoid(m3b9,
                        TimeVars([delta, omega, Ux, Uy, Uxg, Uyg, Ixg, Iyg],
                                 length=int(T / dt) + 1),
                        dt=dt, T=T, event=e1)

# plt.figure(1)
# labels = []
# for i in range(3):
#     plt.plot(np.arange(0, T + dt, dt), xy.array[i + 3, :].reshape((-1,)))
#     labels.append(r'$\omega_%i$' % i)
# plt.legend(labels)
#
# plt.figure(2)
# labels = []
# for i in range(9):
#     plt.plot(np.arange(0, T + dt, dt),
#              (xy.array[i + 6, :].reshape((-1,)) ** 2 + xy.array[i + 15, :].reshape((-1,)) ** 2) ** (1 / 2))
#     labels.append(r'$U_%i$' % i)
# plt.legend(labels)
#
# plt.figure(3)
# labels = []
# for i in range(3):
#     plt.plot(np.arange(0, T + dt, dt),
#              (xy['Ixg'][i, :]))
#     labels.append(r'$I_{xg%i}$' % i)
# plt.legend(labels)
#
# plt.figure(4)
# labels = []
# for i in range(3):
#     plt.plot(np.arange(0, T + dt, dt),
#              (xy['Iyg'][i, :]))
#     labels.append(r'$I_{yg%i}$' % i)
# plt.legend(labels)
#
# plt.show()


def test_m3b9():
    assert np.abs(np.asarray(df['omega']).transpose() - xy['omega'].array).max() <= 0.003969440114605538
    assert np.abs(np.asarray(df['delta']).transpose() - xy['delta'].array).max() <= 0.019164859675127266
