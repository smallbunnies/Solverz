from functools import partial

import numpy as np
import pandas as pd

from Solverz import idx, Var, as_Vars, Eqn, DAE, Rodas, Param, HyperbolicPde, minmod_flag, IdxParam, made_numerical, \
    Rodas_numerical, Opt, parse_dae_v

# %% non-periodical boundary condition
# 1st-order scheme
T = 2
L = 2 * np.pi
Y1_np = []  # results 1st-order non-periodical
for ndx in [100, 200]:
    dx = L / ndx

    x = np.linspace(0 - dx / 2, L + dx / 2, ndx + 2)
    u0 = np.sin(x)
    u = Var('u', value=u0)
    uL = Var('uL', value=0)
    uR = Var('uR', value=0)

    pde = HyperbolicPde("Inviscid burger's equation",
                        diff_var=u,
                        flux=u ** 2 / 2,
                        two_dim_var=u)

    dae0 = pde.semi_discretize(scheme=2)

    aeul = Eqn('equation of ul',
               uR - uL)
    aeur = Eqn('equation of ur',
               uL)

    dae = DAE(dae0 + [aeul, aeur])
    dae.param_initializer('M', IdxParam('M', value=np.array([ndx + 1])))


    def max_func1(u, M1):
        return np.maximum(np.abs(u[2:M1 + 1]), np.abs(u[1:M1]))


    def max_func2(u, M1):
        return np.maximum(np.abs(np.abs(u[1:M1])), np.abs(u[0:M1 - 1]))


    dae.param_initializer('ajp12',
                          Param(name='ajp12',
                                triggerable=True,
                                trigger_var='u',
                                trigger_fun=partial(max_func1,
                                                    M1=ndx + 1)))
    dae.param_initializer('ajm12',
                          Param(name='ajm12',
                                triggerable=True,
                                trigger_var='u',
                                trigger_fun=partial(max_func2,
                                                    M1=ndx + 1)))

    dae.param_initializer('dx',
                          Param(name='dx', value=np.array([L / ndx])))

    uL = Var('uL', value=0)
    uR = Var('uR', value=0)
    Tp, Y1, stats1 = Rodas(dae,
                           np.linspace(0, T, 201),
                           as_Vars([u, uR, uL]))
    Y1_np.append(Y1)

# 2nd-order scheme
# Y2_np = []  # results 2nd-order non-periodical
# for ndx in [100, 200]:
#     dx = L / ndx
#
#     x = np.linspace(0 - dx / 2, L + dx / 2, ndx + 2)
#     u0 = np.sin(x)
#     u = Var('u', value=u0)
#
#     pde = HyperbolicPde("Inviscid burger's equation",
#                         diff_var=u,
#                         flux=u ** 2 / 2,
#                         two_dim_var=u)
#
#     dae0 = pde.semi_discretize(scheme=1, M=ndx+1)
#
#     uL = Var('uL', value=0)
#     uR = Var('uR', value=0)
#
#     aeul = Eqn('equation of ul',
#                uR - uL)
#     aeur = Eqn('equation of ur',
#                uL)
#
#     dae = DAE(dae0 + [aeul, aeur])
#     # dae.param_initializer('M', Param('M', value=np.array([ndx + 1])))
#
#
#     def max_func1(u, M1):
#         return np.maximum(np.abs(u[2:M1 + 1]), np.abs(u[1:M1]))
#
#
#     def max_func2(u, M1):
#         return np.maximum(np.abs(np.abs(u[1:M1])), np.abs(u[0:M1 - 1]))
#
#
#     dae.param_initializer('ajp12',
#                           Param(name='ajp12',
#                                 value=max_func1(u0, ndx+1),
#                                 triggerable=True,
#                                 trigger_var='u',
#                                 trigger_fun=partial(max_func1,
#                                                     M1=ndx + 1)))
#     dae.param_initializer('ajm12',
#                           Param(name='ajm12',
#                                 value=max_func2(u0, ndx+1),
#                                 triggerable=True,
#                                 trigger_var='u',
#                                 trigger_fun=partial(max_func2,
#                                                     M1=ndx + 1)))
#
#
#     def minmod_func(u, theta1, dx1, M1):
#         return minmod_flag(theta1 * (-u[0:M1 - 1] + u[1:M1]) / dx1,
#                            (-u[0:M1 - 1] + u[2:M1 + 1]) / (2 * dx1),
#                            theta1 * (-u[1:M1] + u[2:M1 + 1]) / dx1)
#
#
#     dae.param_initializer('minmod_flag_of_ux',
#                           Param(name='minmod_flag_of_ux',
#                                 value=minmod_func(u0, 1, dx, ndx+1),
#                                 triggerable=True,
#                                 trigger_var='u',
#                                 trigger_fun=partial(minmod_func,
#                                                     theta1=1,
#                                                     dx1=dx,
#                                                     M1=ndx + 1)))
#     dae.param_initializer('dx',
#                           Param(name='dx', value=np.array([L / ndx])))
#
#     dae.update_param('theta', 1)
#
#     u = Var('u', value=u0)
#     uL = Var('uL', value=0)
#     uR = Var('uR', value=0)
#     ux = Var('ux', value=np.zeros_like(u0))
#     # for consistent initial value
#     Y0 = as_Vars([u, uR, uL, ux])
#     ux0 = np.concatenate([np.array([0]), -dae.g(None, Y0, eqn='minmod limiter 1 of u'), np.array([0])])
#     ux = Var('ux', value=ux0)
#
#     Tp, Y2, stats2 = Rodas(dae,
#                            np.linspace(0, 2, 201),
#                            as_Vars([u, uR, uL, ux]))
#     Y2_np.append(Y2)
#
# bench = pd.read_excel('instances/test_burger.xlsx',
#                       sheet_name=None,
#                       engine='openpyxl'
#                       )
#
# bench_ndx100 = np.asarray(bench['np_ndx100']['u'])
# bench_ndx200 = np.asarray(bench['np_ndx200']['u'])
#
# u1_ndx100 = Y1_np[0]['u'][-1, 1:101]
# u2_ndx100 = Y2_np[0]['u'][-1, 1:101]
# u1_ndx200 = Y1_np[1]['u'][-1, 1:201]
# u2_ndx200 = Y2_np[1]['u'][-1, 1:201]
#
#
# def test_burger1():
#     assert np.mean(np.abs(u1_ndx100 - bench_ndx100)) <= 0.016610
#     assert np.mean(np.abs(u2_ndx100 - bench_ndx100)) <= 0.008201
#     assert np.mean(np.abs(u1_ndx200 - bench_ndx200)) <= 0.008336
#     assert np.mean(np.abs(u2_ndx200 - bench_ndx200)) <= 0.003996

# # periodical boundary condition
# # TODO: DERIVE THE EXACT SOLUTIONS AND COMPARE THE RESULTS.
# # 1st-order scheme
# T = 2
# L = 2 * np.pi
# Y1_p = []  # results 1st-order non-periodical
# for ndx in [100, 200]:
#     dx = L / ndx
#
#     x = np.linspace(0 - dx / 2, L + dx / 2, ndx + 2)
#     u0 = np.sin(x) + 0.5
#     u = Var('u', value=u0)
#
#     pde = HyperbolicPde("Inviscid burger's equation",
#                         diff_var=u,
#                         flux=u ** 2 / 2,
#                         two_dim_var=u)
#
#     dae0 = pde.semi_discretize(scheme=2)
#
#     uL = Var('uL')
#     uR = Var('uR')
#
#     aeul = Eqn('equation of ul',
#                uR - uL)
#     aeur = Eqn('equation of ur',
#                uL - 3 / 2 * u[1] + 1 / 2 * u[2])
#
#     dae = DAE(dae0 + [aeul, aeur])
#     dae.param_initializer('M', Param('M', value=np.array([ndx + 1])))
#
#
#     def max_func1(u, M1):
#         return np.maximum(np.abs(u[2:M1 + 1]), np.abs(u[1:M1]))
#
#
#     def max_func2(u, M1):
#         return np.maximum(np.abs(np.abs(u[1:M1])), np.abs(u[0:M1 - 1]))
#
#
#     dae.param_initializer('ajp12',
#                           Param(name='ajp12',
#                                 triggerable=True,
#                                 trigger_var='u',
#                                 trigger_fun=partial(max_func1,
#                                                     M1=ndx + 1)))
#     dae.param_initializer('ajm12',
#                           Param(name='ajm12',
#                                 triggerable=True,
#                                 trigger_var='u',
#                                 trigger_fun=partial(max_func2,
#                                                     M1=ndx + 1)))
#
#     dae.param_initializer('dx',
#                           Param(name='dx', value=np.array([L / ndx])))
#
#     uL = Var('uL', value=0.5)
#     uR = Var('uR', value=0.5)
#     Tp, Y1, stats1 = Rodas(dae,
#                            np.linspace(0, T, 201),
#                            as_Vars([u, uR, uL]))
#     Y1_p.append(Y1)
#
# # 2nd-order scheme
# Y2_p = []  # results 2nd-order non-periodical
# for ndx in [100, 200]:
#     dx = L / ndx
#
#     x = np.linspace(0 - dx / 2, L + dx / 2, ndx + 2)
#     u0 = np.sin(x) + 0.5
#     u = Var('u', value=u0)
#
#     pde = HyperbolicPde("Inviscid burger's equation",
#                         diff_var=u,
#                         flux=u ** 2 / 2,
#                         two_dim_var=u)
#
#     dae0 = pde.semi_discretize(scheme=1)
#
#     uL = Var('uL', value=0.5)
#     uR = Var('uR', value=0.5)
#
#     aeul = Eqn('equation of ul',
#                uR - uL)
#     aeur = Eqn('equation of ur',
#                uL - 3 / 2 * u[1] + 1 / 2 * u[2])
#
#     dae = DAE(dae0 + [aeul, aeur])
#     dae.param_initializer('M', Param('M', value=np.array([ndx + 1])))
#
#
#     def max_func1(u, M1):
#         return np.maximum(np.abs(u[2:M1 + 1]), np.abs(u[1:M1]))
#
#
#     def max_func2(u, M1):
#         return np.maximum(np.abs(np.abs(u[1:M1])), np.abs(u[0:M1 - 1]))
#
#
#     dae.param_initializer('ajp12',
#                           Param(name='ajp12',
#                                 triggerable=True,
#                                 trigger_var='u',
#                                 trigger_fun=partial(max_func1,
#                                                     M1=ndx + 1)))
#     dae.param_initializer('ajm12',
#                           Param(name='ajm12',
#                                 triggerable=True,
#                                 trigger_var='u',
#                                 trigger_fun=partial(max_func2,
#                                                     M1=ndx + 1)))
#
#
#     def minmod_func(u, theta1, dx1, M1):
#         return minmod_flag(theta1 * (-u[0:M1 - 1] + u[1:M1]) / dx1,
#                            (-u[0:M1 - 1] + u[2:M1 + 1]) / (2 * dx1),
#                            theta1 * (-u[1:M1] + u[2:M1 + 1]) / dx1)
#
#
#     dae.param_initializer('minmod_flag_of_ux',
#                           Param(name='minmod_flag_of_ux',
#                                 triggerable=True,
#                                 trigger_var='u',
#                                 trigger_fun=partial(minmod_func,
#                                                     theta1=1,
#                                                     dx1=dx,
#                                                     M1=ndx + 1)))
#     dae.param_initializer('dx',
#                           Param(name='dx', value=np.array([L / ndx])))
#
#     dae.update_param('theta', 1)
#
#     u = Var('u', value=u0)
#     uL = Var('uL', value=0.5)
#     uR = Var('uR', value=0.5)
#     ux = Var('ux', value=np.zeros_like(u0))
#
#     # for consistent initial value
#     Y0 = as_Vars([u, uR, uL, ux])
#     ux0 = np.concatenate([np.array([0]), -dae.g(Y0, eqn='minmod limiter 1 of u'), np.array([0])])
#     ux = Var('ux', value=ux0)
#
#     Tp, Y2, stats2 = Rodas(dae,
#                            np.linspace(0, 2, 201),
#                            as_Vars([u, uR, uL, ux]))
#     Y2_p.append(Y2)
#
# benchp1_ndx100 = np.asarray(bench['p_ndx100']['u1'])
# benchp2_ndx100 = np.asarray(bench['p_ndx100']['u2'])
# benchp1_ndx200 = np.asarray(bench['p_ndx200']['u1'])
# benchp2_ndx200 = np.asarray(bench['p_ndx200']['u2'])
#
# u1p_ndx100 = Y1_p[0]['u'][-1, :]
# u2p_ndx100 = Y2_p[0]['u'][-1, :]
# u1p_ndx200 = Y1_p[1]['u'][-1, :]
# u2p_ndx200 = Y2_p[1]['u'][-1, :]
#
#
# def test_burger2():
#     assert np.max(np.abs(u1p_ndx100 - benchp1_ndx100)) <= 1e-7
#     assert np.max(np.abs(u2p_ndx100 - benchp2_ndx100)) <= 1e-7
#     assert np.max(np.abs(u1p_ndx200 - benchp1_ndx200)) <= 1e-7
#     assert np.max(np.abs(u2p_ndx200 - benchp2_ndx200)) <= 1e-7
