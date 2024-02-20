import pandas as pd
import numpy as np
from functools import partial

from Solverz import Eqn, AE, nr_method, as_Vars, Var, Param, continuous_nr, Para, idx, Abs, exp, Mat_Mul, \
    made_numerical

# %% initialize variables and params
sys_df = pd.read_excel('instances/4node3pipe_change_sign.xlsx',
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=0
                       )


def derive_incidence_matrix(f_node, t_node):
    V = np.zeros((max(max(f_node), max(t_node)) + 1, len(f_node)))
    for pipe in range(len(f_node)):
        V[f_node[pipe], pipe] = -1
        V[t_node[pipe], pipe] = 1
    return V


def derive_v_plus(v: np.ndarray) -> np.ndarray:
    return (v + np.abs(v)) / 2


def derive_v_minus(v: np.ndarray) -> np.ndarray:
    return (v - np.abs(v)) / 2


def v_p_reverse_pipe(V0: np.ndarray, m0: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Trigger function of v_p
    if some element of m changes its sign, then related column of V0 changes its sign
    :param V0:
    :param m0: initial mass flow rate
    :param m:
    :return: $V_0*(I-diag(sign(m0)-sign(m)))$
    """
    # pipe i changes sign then m_sign[i]=2
    m_sign = np.abs(np.sign(m0) - np.sign(m))
    return derive_v_plus(V0 @ (np.eye(m.shape[0]) - np.diagflat(m_sign)))


def v_m_reverse_pipe(V0: np.ndarray, m0: np.ndarray, m: np.ndarray) -> np.ndarray:
    # pipe i changes sign then m_sign[i]=2
    m_sign = np.abs(np.sign(m0) - np.sign(m))
    return derive_v_minus(V0 @ (np.eye(m.shape[0]) - np.diagflat(np.abs(m_sign))))


def f_node_calculator(f_node0: np.ndarray, t_node0: np.ndarray, m0: np.ndarray, m: np.ndarray) -> np.ndarray:
    # pipe i changes sign then m_sign[i]=2
    m_sign = np.abs(np.sign(m0) - np.sign(m))
    flag_change_sign = m_sign > 0
    return np.where(flag_change_sign.reshape((-1,)), t_node0, f_node0)


def t_node_calculator(f_node0: np.ndarray, t_node0: np.ndarray, m0: np.ndarray, m: np.ndarray) -> np.ndarray:
    # pipe i changes sign then m_sign[i]=2
    m_sign = np.abs(np.sign(m0) - np.sign(m))
    flag_change_sign = m_sign > 0
    return np.where(flag_change_sign.reshape((-1,)), f_node0, t_node0)


# %% md
# Declare equations and parameters
# %%
Node = sys_df['Node']
i = idx(name='i', value=np.asarray(Node['type'][Node['type'] == 3].index))  # intermediate nodes
l = idx(name='l', value=np.asarray(Node['type'][Node['type'] == 2].index))  # load nodes
s = idx(name='s', value=np.asarray(Node['type'][Node['type'] == 1].index))  # non-balance source nodes
r = idx(name='r', value=np.asarray(Node['type'][Node['type'] == 0].index))  # balance source nodes
sl = idx(name='sl', value=np.concatenate((s.value, l.value)))
rs = idx(name='rs', value=np.concatenate((r.value, s.value)))
rsl = idx(name='rsl', value=np.concatenate((r.value, s.value, l.value)))
li = idx(name='li', value=np.concatenate((l.value, i.value)))
rsi = idx(name='rsi', value=np.concatenate((r.value, s.value, i.value)))

mL = Para('mL', dim=2, value=np.asarray(sys_df['m_L']))
K = Para(name='K', value=np.asarray(sys_df['K']).reshape(-1, ))
phi_set = Para(name='phi_set', value=np.asarray(sys_df['phi_set']).reshape(-1, ))

Cp = Para(name='Cp', value=np.asarray(sys_df['Cp']).reshape(-1, ))
L = Para(name='L', value=np.asarray(sys_df['L']).reshape(-1, ))
coeff_lambda = Para(name='coeff_lambda', value=np.asarray(sys_df['coeff_lambda']).reshape(-1, ))
Ta = Para(name='Ta', value=np.asarray(sys_df['Ta']).reshape(-1, ))
f_node = sys_df['Pipe']['f_node']
t_node = sys_df['Pipe']['t_node']
V = Para(name='V', dim=2, value=derive_incidence_matrix(f_node, t_node))
Vp = Para(name='Vp', dim=2, value=derive_v_plus(V.value))
Vm = Para(name='Vm', dim=2, value=derive_v_minus(V.value))
f_node = idx(name='f_node', value=np.asarray(f_node).reshape(-1, ))  # intermediate nodes
t_node = idx(name='t_node', value=np.asarray(t_node).reshape(-1, ))  # load nodes

m = Var(name='m', value=np.asarray(sys_df['var']['m']))
mq = Var(name='mq', value=np.asarray(sys_df['var']['mq']))
Ts = Var(name='Ts', value=np.asarray(sys_df['var']['Ts']))
Tr = Var(name='Tr', value=np.asarray(sys_df['var']['Tr']))
Touts = Var(name='Touts', value=np.asarray(sys_df['var']['Touts']))
Toutr = Var(name='Toutr', value=np.asarray(sys_df['var']['Toutr']))
phi = Var(name='phi', value=np.asarray(sys_df['var']['phi']))

E1 = Eqn(name='E1', eqn=(Ts[f_node] - Ta) * exp(-coeff_lambda * L / (Cp * Abs(m))) + Ta - Touts)
E3 = Eqn(name='E3', eqn=(Tr[t_node] - Ta) * exp(-coeff_lambda * L / (Cp * Abs(m))) + Ta - Toutr)
E5 = Eqn(name='E5', eqn=Mat_Mul(V[rs, :], m) + mq[rs])
E6 = Eqn(name='E6', eqn=Mat_Mul(V[l, :], m) - mq[l])
E7 = Eqn(name='E7', eqn=Mat_Mul(V[i, :], m))
E8 = Eqn(name='E8', eqn=Mat_Mul(mL, K * Abs(m) * m))
E9 = Eqn(name='E9', eqn=Ts[li] * Mat_Mul(Vp[li, :], Abs(m)) - Mat_Mul(Vp[li, :], Touts * Abs(m)))
E10 = Eqn(name='E10', eqn=Tr[rsi] * Mat_Mul(Vm[rsi, :], Abs(m)) - Mat_Mul(Vm[rsi, :], Toutr * Abs(m)))
E11 = Eqn(name='E11', eqn=phi[rsl] - 4182 * mq[rsl] * (Ts[rsl] - Tr[rsl]))
E12 = Eqn(name='E12', eqn=Ts[rs] - 100)
E13 = Eqn(name='E13', eqn=Tr[l] - 50)
E14 = Eqn(name='E14', eqn=phi[sl] - phi_set)
E15 = Eqn(name='E15', eqn=phi[i])
E16 = Eqn(name='E16', eqn=mq[i])
E = AE(name='Pipe Equations', eqn=[E1, E3, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16])

E.param_initializer('Vp', param=Param('Vp',
                                      value=Vp.value,
                                      triggerable=True,
                                      trigger_var='m',
                                      trigger_fun=partial(v_p_reverse_pipe, V.value, m.value),
                                      dim=2
                                      ))

E.param_initializer('Vm', param=Param('Vm',
                                      value=Vm.value,
                                      triggerable=True,
                                      trigger_var='m',
                                      trigger_fun=partial(v_m_reverse_pipe, V.value, m.value),
                                      dim=2
                                      ))

E.param_initializer('f_node', param=Param('f_node',
                                          value=f_node.value,
                                          triggerable=True,
                                          trigger_var='m',
                                          trigger_fun=partial(f_node_calculator,
                                                              f_node.value, t_node.value, m.value),
                                          dtype=int
                                          ))

E.param_initializer('t_node', param=Param('t_node',
                                          value=t_node.value,
                                          triggerable=True,
                                          trigger_var='m',
                                          trigger_fun=partial(t_node_calculator,
                                                              f_node.value, t_node.value, m.value),
                                          dtype=int
                                          ))

y0 = as_Vars([m, mq, Ts, Tr, Touts, Toutr, phi])
nE, code = made_numerical(E, y0, output_code=True)

ynr = nr_method(nE, y0)
# y_cnr, ite = continuous_nr(nE, y0.array)


sys_df = pd.read_excel('instances/4node3pipe_change_sign_bench.xlsx',
                       sheet_name=None,
                       engine='openpyxl',
                       header=None)


def test_nr_method():
    for var_name in ['Ts', 'Tr', 'm', 'mq', 'phi']:
        # find nonzero elements
        idx_nonzero = np.nonzero(ynr[var_name])
        assert max(abs((ynr[var_name][idx_nonzero] - np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, ))) /
                   np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, )) <= 1e-8


# def test_cnr_method():
#     for var_name in ['Ts', 'Tr', 'm', 'mq', 'phi']:
#         # find nonzero elements
#         idx_nonzero = np.nonzero(y_cnr[var_name])
#         assert max(abs((y_cnr[var_name][idx_nonzero] - np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, )) /
#                        np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, ))) <= 1e-8
