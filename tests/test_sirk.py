import numpy as np
from numpy import linalg
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from core.algebra import AliasVar, ComputeParam, F, X, Y
from core.eqn import Ode, Eqn
from core.equations import DAE
from core.param import Param
from core.solver import nr_method, sirk_dae, sirk_ode
from core.var import Var, TimeVar
from core.variables import Vars, TimeVars
from core.solverz_array import SolverzArray

# van der pol equation

vdp1_1 = Ode(name='vdp1_1', e_str='x2', diff_var='x1')
vdp1_2 = Ode(name='vdp1_2', e_str='(1-x1**2)*x2-x1', diff_var='x2')
vdp1 = DAE([vdp1_1, vdp1_2], name='vdp1')
x1 = TimeVar('x1')
x1.v = [2]
x2 = TimeVar('x2')
x2.v = [0]

T = 10
dt = 0.01
x = TimeVars([x1, x2], length=int(T / dt) + 1)

# x = sirk_ode(vdp1, x, T, dt)
# plt.plot(np.arange(0, T + dt, dt), x.array[0, :])

# a simple dae

f = Ode(name='f', e_str='-x**3+0.5*y**2', diff_var='x')
g = Eqn(name='g', e_str='x**2+y**2-2')
dae = DAE([f, g])
x = TimeVar('x')
x.v = [1]
y = TimeVar('y')
y.v = [1]

T = 10
dt = 0.01
x = TimeVars([x], length=int(T / dt) + 1)
y = TimeVars([y], length=int(T / dt) + 1)

# x, y = sirk_dae(dae, x, y, T, dt)
#
# plt.figure(1)
# plt.plot(np.arange(0, T + dt, dt), x.array[0, :])
#
# plt.figure(2)
# plt.plot(np.arange(0, T + dt, dt), y.array[0, :])

# m3b9

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

x = TimeVars([omega, delta], length=int(T / dt) + 1)
y = TimeVars([Ux, Uy, Uxg, Uyg, Ixg, Iyg], length=int(T / dt) + 1)

dt = np.array(0.003)
T = np.array(3)


def sirk_dae1(dae: DAE,
              x: TimeVars,
              y: TimeVars,
              T,
              dt):
    r = 0.572816
    r21 = -2.34199281090742
    r31 = -0.0273333497228473
    r32 = 0.213811780334800
    r41 = -0.259083734468120
    r42 = -0.190595825778459
    r43 = -0.228030947223683
    a21 = 1.145632
    a31 = 0.520920769237066
    a32 = 0.134294177986617
    a41 = 0.520920769237066
    a42 = 0.134294177986617
    a43 = 0
    b1 = 0.324534692057929
    b2 = 0.0490865790926829
    b3 = 0
    b4 = 0.626378728849388

    # a21 = 1
    # a31 = 1
    # a32 = 0
    # r21 = (1 - 6 * r) / 2
    # r31 = (12 * r ** 2 - 6 * r - 1) / (3 * (1 - 2 * r))
    # r32 = (12 * r ** 2 - 12 * r + 5) / (6 * (1 - 2 * r))
    # b1 = 2 / 3
    # b2 = 1 / 6
    # b3 = 1 / 6

    dae.assign_eqn_var_address(x[0], y[0])
    block_I = np.zeros((dae.var_size, dae.var_size))
    block_I[np.ix_(range(dae.state_num), range(dae.state_num))] = block_I[np.ix_(range(dae.state_num),
                                                                                 range(dae.state_num))] + np.eye(
        dae.state_num)

    for i in range(int(T / dt)):

        if i*dt >= 0.000:
            G = np.asarray(df['G'])
            G[6, 6] = 10000
            dae.update_param('G', G)
        if i*dt >= 0.030:
            G = np.asarray(df['G'])
            dae.update_param('G', G)

        J0 = dae.j(x[i], y[i])
        J_inv = SolverzArray(linalg.inv(block_I - dt * J0 * r)) * dt
        k1 = J_inv * (np.concatenate((dae.f(x[i], y[i]), dae.g(x[i], y[i])), axis=0))
        k2 = J_inv * (np.concatenate(
            (dae.f(x[i] + a21 * k1[0:dae.state_num], y[i] + a21 * k1[dae.state_num:dae.var_size]),
             dae.g(x[i] + a21 * k1[0:dae.state_num], y[i] + a21 * k1[dae.state_num:dae.var_size])), axis=0)
                      + J0 * r21 * k1)
        k3 = J_inv * (np.concatenate((dae.f(x[i] + a31 * k1[0:dae.state_num] + a32 * k2[0:dae.state_num],
                                            y[i] + a31 * k1[dae.state_num:dae.var_size] + a32 * k2[
                                                                                                dae.state_num:dae.var_size]),
                                      dae.g(x[i] + a31 * k1[0:dae.state_num] + a32 * k2[0:dae.state_num],
                                            y[i] + a31 * k1[dae.state_num:dae.var_size] + a32 * k2[
                                                                                                dae.state_num:dae.var_size])),
                                     axis=0) +
                      J0 * (r31 * k1 + r32 * k2))
        k4 = J_inv * (
                np.concatenate(
                    (dae.f(x[i] + a41 * k1[0:dae.state_num] + a42 * k2[0:dae.state_num] + a43 * k3[0:dae.state_num],
                           y[i] + a41 * k1[dae.state_num:dae.var_size] + a42 * k2[
                                                                               dae.state_num:dae.var_size] + a43 * k3[
                                                                                                                   dae.state_num:dae.var_size]),
                     dae.g(x[i] + a41 * k1[0:dae.state_num] + a42 * k2[0:dae.state_num] + a43 * k3[0:dae.state_num],
                           y[i] + a41 * k1[dae.state_num:dae.var_size] + a42 * k2[
                                                                               dae.state_num:dae.var_size] + a43 * k3[
                                                                                                                   dae.state_num:dae.var_size])),
                    axis=0) +
                J0 * (r41 * k1 + r42 * k2 + r43 * k3))
        x[i + 1] = x[i] + b1 * k1[0:dae.state_num] + b2 * k2[0:dae.state_num] + b3 * k3[0:dae.state_num] + b4 * k4[
                                                                                                                0:dae.state_num]
        y[i + 1] = y[i] + b1 * k1[dae.state_num:dae.var_size] + b2 * k2[dae.state_num:dae.var_size] + b3 * k3[
                                                                                                           dae.state_num:dae.var_size] + b4 * k4[
                                                                                                                                              dae.state_num:dae.var_size]

    return x, y


x, y = sirk_dae1(m3b9, x, y, T, dt)

plt.figure(1)
labels = []
for i in range(3):
    plt.plot(np.arange(0, T + dt, dt), x['omega'][i].reshape((-1,)))
    labels.append(r'$\omega_%i$' % i)
plt.legend(labels)

plt.figure(2)
labels = []
for i in range(3):
    plt.plot(np.arange(0, T + dt, dt), x['delta'][i].reshape((-1,)))
    labels.append(r'$\delta_%i$' % i)
plt.legend(labels)

plt.figure(3)
labels = []
for i in range(9):
    plt.plot(np.arange(0, T + dt, dt), (y['Ux'][i].reshape((-1,))**2+y['Uy'][i].reshape((-1,))**2)**(1/2))
    labels.append(r'$U_%i$' % i)
plt.legend(labels)


plt.figure(4)
labels = []
for i in range(30):
    plt.plot(np.arange(0, T + dt, dt), )
    labels.append(r'$\Delta_%i$' % i)
plt.legend(labels)
