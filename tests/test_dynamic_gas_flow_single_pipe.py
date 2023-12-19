# import numpy as np
# import pandas as pd
#
# from Solverz import idx, Var, Param_, as_Vars, Const_, HyperbolicPde, AE, fde_solver, Eqn, Param
#
# results = pd.read_excel('instances/dynamic_gas_flow_single_pipe.xlsx',
#                         sheet_name=None,
#                         engine='openpyxl'
#                         )
#
# p = Var('p', value=np.linspace(9, 8.728, 11) * 1e6)
# q = Var('q', value=7.3125 * np.ones((11, 1)))
# va = Const_('va', value=340)
# D = Const_('D', value=0.5)
# S = Const_('S', value=np.pi * (D.value / 2) ** 2)
# lam = Const_('lam', 0.3)
# pb = Param_('pb', 9e6)
# qb = Param_('qb', 7.3125)
# M = idx('M')
# L = 3000
#
# gas_pde1 = HyperbolicPde('mass conservation',
#                          diff_var=p,
#                          flux=va ** 2 / S * q,
#                          two_dim_var=[p, q])
# gas_pde2 = HyperbolicPde('momentum conservation',
#                          diff_var=q,
#                          flux=S * p,
#                          source=-lam * va ** 2 * q ** 2 / (2 * D * S * p),
#                          two_dim_var=[p, q])
#
# ae1 = gas_pde1.finite_difference()
# ae2 = gas_pde2.finite_difference()
# ae3 = Eqn(name='boundary condition p', eqn=p[0] - pb)
# ae4 = Eqn(name='boundary condition q', eqn=q[M] - qb)
#
# gas_FDE = AE([ae1, ae2, ae3, ae4], name='gas FDE')
# u0 = as_Vars([p, q])
#
# T = 3600
# gas_FDE.param_initializer('M', Param('M', value=np.array(10)))
# gas_FDE.param_initializer('dx', Param('dx', value=np.array(300)))
# u = fde_solver(gas_FDE, u0, [0, 3600], 5, tol=1e-3)
#
#
# def test_fde_solver():
#     pout = np.asarray(results['results']['pout'])
#     qin = np.asarray(results['results']['qin'])
#
#     # if matrix container is cvxopt.spmatrix
#     # assert np.mean(abs(pout - u['p'][:, -1] / 1e6)) < 5e-6
#     # assert np.mean(abs(qin - u['q'][:, 0])) < 1.2e-4
#
#     # if matrix container is scipy.sparse.csc_array
#     assert np.mean(abs(pout - u['p'][:, -1] / 1e6)) < 1e-10
#     assert np.mean(abs(qin - u['q'][:, 0])) < 1e-7
