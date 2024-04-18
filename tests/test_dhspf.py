# import pandas as pd
# import numpy as np
#
# from Solverz import Eqn, AE, nr_method, as_Vars, iVar, Para, idx, Abs, exp, Mat_Mul, made_numerical, Param, IdxParam
#
# # %% initialize variables and params
# sys_df = pd.read_excel('instances/4node3pipe.xlsx',
#                        sheet_name=None,
#                        engine='openpyxl',
#                        index_col=0
#                        )
#
#
# def derive_incidence_matrix(f_node, t_node):
#     V = np.zeros((max(max(f_node), max(t_node)) + 1, len(f_node)))
#     for pipe in range(len(f_node)):
#         V[f_node[pipe], pipe] = -1
#         V[t_node[pipe], pipe] = 1
#     return V
#
#
# def derive_v_plus(v: np.ndarray) -> np.ndarray:
#     return (v + np.abs(v)) / 2
#
#
# def derive_v_minus(v: np.ndarray) -> np.ndarray:
#     return (v - np.abs(v)) / 2
#
#
# # %% md
# # Declare equations and parameters
# # %%
# Node = sys_df['Node']
# i = idx(name='i', value=np.asarray(Node['type'][Node['type'] == 3].index))  # intermediate nodes
# l = idx(name='l', value=np.asarray(Node['type'][Node['type'] == 2].index))  # load nodes
# s = idx(name='s', value=np.asarray(Node['type'][Node['type'] == 1].index))  # non-balance source nodes
# r = idx(name='r', value=np.asarray(Node['type'][Node['type'] == 0].index))  # balance source nodes
# sl = idx(name='sl', value=np.concatenate((s.value, l.value)))
# rs = idx(name='rs', value=np.concatenate((r.value, s.value)))
# rsl = idx(name='rsl', value=np.concatenate((r.value, s.value, l.value)))
# li = idx(name='li', value=np.concatenate((l.value, i.value)))
# rsi = idx(name='rsi', value=np.concatenate((r.value, s.value, i.value)))
#
# mL = Para('mL', dim=2, value=np.asarray(sys_df['m_L']))
# K = Para(name='K', value=np.asarray(sys_df['K']).reshape(-1, ))
# Ts_set = Para(name='Ts_set', value=np.asarray(sys_df['Ts_set']).reshape(-1, ))
# Tr_set = Para(name='Tr_set', value=np.asarray(sys_df['Tr_set']).reshape(-1, ))
# phi_set = Para(name='phi_set', value=np.asarray(sys_df['phi_set']).reshape(-1, ))
#
# Cp = Para(name='Cp', value=np.asarray(sys_df['Cp']).reshape(-1, ))
# L = Para(name='L', value=np.asarray(sys_df['L']).reshape(-1, ))
# coeff_lambda = Para(name='coeff_lambda', value=np.asarray(sys_df['coeff_lambda']).reshape(-1, ))
# Ta = Para(name='Ta', value=np.asarray(sys_df['Ta']).reshape(-1, ))
# f_node = sys_df['Pipe']['f_node']
# t_node = sys_df['Pipe']['t_node']
# V = Para(name='V', dim=2, value=derive_incidence_matrix(f_node, t_node))
# Vp = Para(name='Vp', dim=2, value=derive_v_plus(V.value))
# Vm = Para(name='Vm', dim=2, value=derive_v_minus(V.value))
# f_node = idx(name='f_node', value=np.asarray(f_node).reshape(-1, ))  # intermediate nodes
# t_node = idx(name='t_node', value=np.asarray(t_node).reshape(-1, ))  # load nodes
#
# m = iVar(name='m', value=np.asarray(sys_df['var']['m']))
# mq = iVar(name='mq', value=np.asarray(sys_df['var']['mq']))
# Ts = iVar(name='Ts', value=np.asarray(sys_df['var']['Ts']))
# Tr = iVar(name='Tr', value=np.asarray(sys_df['var']['Tr']))
# Touts = iVar(name='Touts', value=np.asarray(sys_df['var']['Touts']))
# Toutr = iVar(name='Toutr', value=np.asarray(sys_df['var']['Toutr']))
# phi = iVar(name='phi', value=np.asarray(sys_df['var']['phi']))
#
# E1 = Eqn(name='E1', eqn=(Ts[f_node] - Ta) * exp(-coeff_lambda * L / (Cp * Abs(m))) + Ta - Touts)
# E3 = Eqn(name='E3', eqn=(Tr[t_node] - Ta) * exp(-coeff_lambda * L / (Cp * Abs(m))) + Ta - Toutr)
# E5 = Eqn(name='E5', eqn=Mat_Mul(V[rs, :], m) + mq[rs])
# E6 = Eqn(name='E6', eqn=Mat_Mul(V[l, :], m) - mq[l])
# E7 = Eqn(name='E7', eqn=Mat_Mul(V[i, :], m))
# E8 = Eqn(name='E8', eqn=Mat_Mul(mL, K * Abs(m) * m))
# E9 = Eqn(name='E9', eqn=Ts[li] * Mat_Mul(Vp[li, :], Abs(m)) - Mat_Mul(Vp[li, :], Touts * Abs(m)))
# E10 = Eqn(name='E10', eqn=Tr[rsi] * Mat_Mul(Vm[rsi, :], Abs(m)) - Mat_Mul(Vm[rsi, :], Toutr * Abs(m)))
# E11 = Eqn(name='E11', eqn=phi[rsl] - 4182 * mq[rsl] * (Ts[rsl] - Tr[rsl]))
# E12 = Eqn(name='E12', eqn=Ts[rs] - Ts_set)
# E13 = Eqn(name='E13', eqn=Tr[l] - Tr_set)
# E14 = Eqn(name='E14', eqn=phi[sl] - phi_set)
# E15 = Eqn(name='E15', eqn=phi[i])
# E16 = Eqn(name='E16', eqn=mq[i])
# E = AE(name='Pipe Equations', eqn=[E1, E3, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16])
# y0 = as_Vars([m, mq, Ts, Tr, Touts, Toutr, phi])
#
# nE, code = made_numerical(E, y0, output_code=True)
#
# sol = nr_method(nE, y0)
# # y_cnr, ite = continuous_nr(nE, y0.array)
#
#
# sys_df = pd.read_excel('instances/4node3pipe_bench.xlsx',
#                        sheet_name=None,
#                        engine='openpyxl',
#                        header=None
#                        )
#
#
# def test_nr_method():
#     for var_name in ['Ts', 'Tr', 'm', 'mq', 'phi']:
#         # find nonzero elements
#         idx_nonzero = np.nonzero(sol.y[var_name])
#         assert max(abs((sol.y[var_name][idx_nonzero] - np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, ))) /
#                    np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, )) <= 1e-8
#
# # def test_cnr_method():
# #     for var_name in ['Ts', 'Tr', 'm', 'mq', 'phi']:
# #         # find nonzero elements
# #         idx_nonzero = np.nonzero(y_cnr[var_name])
# #         assert max(abs((y_cnr[var_name][idx_nonzero] - np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, )) /
# #                        np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, ))) <= 1e-8
