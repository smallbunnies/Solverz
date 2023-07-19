import pandas as pd
import numpy as np
from functools import partial

from Solverz import Eqn, AE, nr_method, Var, Vars, Var_, Param, continuous_nr, Param_, Const_, idx, Abs

# %% initialize variables and params
sys_df = pd.read_excel('instances/4node3pipe.xlsx',
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=0
                       )
var_dict = dict()
param_dict = dict()

for varname in sys_df['var'].columns:
    var_dict[varname] = Var(name=varname, value=np.asarray(sys_df['var'][varname]))

param_dict['Cp'] = Param(name='Cp', value=np.asarray(sys_df['Cp']).reshape(-1, ))
param_dict['L'] = Param(name='L', value=np.asarray(sys_df['L']).reshape(-1, ))
param_dict['coeff_lambda'] = Param(name='coeff_lambda', value=np.asarray(sys_df['coeff_lambda']).reshape(-1, ))
param_dict['Ta'] = Param(name='Ta', value=np.asarray(sys_df['Ta']).reshape(-1, ))
param_dict['V_rs'] = Param(name='V_rs', value=np.asarray(sys_df['V_rs']))


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
    m_sign = np.abs(np.sign(m0) - np.sign(m))
    return derive_v_plus(V0 @ (np.eye(m.shape[0]) - np.diag(m_sign)))


def v_m_reverse_pipe(V0: np.ndarray, m0: np.ndarray, m: np.ndarray) -> np.ndarray:
    m_sign = np.abs(np.sign(m0) - np.sign(m))
    return derive_v_minus(V0 @ (np.eye(m.shape[0]) - np.diag(np.abs(m_sign))))


def v_p_li_reverse_pipe(V0: np.ndarray, m0: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Trigger function of v_p
    if some element of m changes its sign, then related column of V0 changes its sign
    :param V0:
    :param m0: initial mass flow rate
    :param m:
    :return: $V_0*(I-diag(sign(m0)-sign(m)))$
    """
    m_sign = np.abs(np.sign(m0) - np.sign(m))
    return derive_v_plus(V0 @ (np.eye(m.shape[0]) - np.diag(m_sign)))


def v_m_rsi_reverse_pipe(V0: np.ndarray, m0: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Trigger function of v_p
    if some element of m changes its sign, then related column of V0 changes its sign
    :param V0:
    :param m0: initial mass flow rate
    :param m:
    :return: $V_0*(I-diag(sign(m0)-sign(m)))$
    """
    m_sign = np.abs(np.sign(m0) - np.sign(m))
    return derive_v_minus(V0 @ (np.eye(m.shape[0]) - np.diag(m_sign)))


param_dict['Vp'] = Param(name='Vp',
                         value=np.asarray(sys_df['Vp']),
                         triggerable=True,
                         trigger_var='m',
                         trigger_fun=partial(v_p_reverse_pipe,
                                             np.asarray(sys_df['Vp']) + np.asarray(sys_df['Vm']),
                                             np.asarray(sys_df['var']['m'])))

param_dict['Vm'] = Param(name='Vm',
                         value=np.asarray(sys_df['Vm']),
                         triggerable=True,
                         trigger_var='m',
                         trigger_fun=partial(v_m_reverse_pipe,
                                             np.asarray(sys_df['Vp']) + np.asarray(sys_df['Vm']),
                                             np.asarray(sys_df['var']['m'])))

param_dict['V_l'] = Param(name='V_l', value=np.asarray(sys_df['V_l']))
param_dict['V_i'] = Param(name='V_i', value=np.asarray(sys_df['V_i']))

param_dict['V_p_li'] = Param(name='V_p_li',
                             value=np.asarray(sys_df['V_p_li']),
                             triggerable=True,
                             trigger_var='m',
                             trigger_fun=partial(v_p_li_reverse_pipe,
                                                 np.asarray(sys_df['V_li0']),
                                                 np.asarray(sys_df['var']['m'])))

param_dict['V_m_rsi'] = Param(name='V_m_rsi',
                              value=np.asarray(sys_df['V_m_rsi']),
                              triggerable=True,
                              trigger_var='m',
                              trigger_fun=partial(v_m_rsi_reverse_pipe,
                                                  np.asarray(sys_df['V_rsi0']),
                                                  np.asarray(sys_df['var']['m'])))

param_dict['A_rsl'] = Param(name='A_rsl', value=np.asarray(sys_df['A_rsl']))
param_dict['A_rs'] = Param(name='A_rs', value=np.asarray(sys_df['A_rs']))
param_dict['A_sl'] = Param(name='A_sl', value=np.asarray(sys_df['A_sl']))
param_dict['A_l'] = Param(name='A_l', value=np.asarray(sys_df['A_l']))
param_dict['A_i'] = Param(name='A_i', value=np.asarray(sys_df['A_i']))
param_dict['A_li'] = Param(name='A_li', value=np.asarray(sys_df['A_li']))
param_dict['A_rsi'] = Param(name='A_rsi', value=np.asarray(sys_df['A_rsi']))
param_dict['Ts_set'] = Param(name='Ts_set', value=[100])
param_dict['Tr_set'] = Param(name='Tr_set', value=[50])
param_dict['phi_set'] = Param(name='phi_set', value=np.asarray(sys_df['phi_set']).reshape(-1, ))

# %% md
# Declare equations and parameters
# %%
mL = Const_('mL', dim=2, value=np.asarray(sys_df['m_L']))
K = Const_(name='K', value=np.asarray(sys_df['K']))
i = idx(name='i', value=[1, 2, 3])
m = Var_(name='m', value=np.asarray(sys_df['var']['m']))
mq = Var_(name='m', value=np.asarray(sys_df['var']['mq']))

E1 = Eqn(name='E1', eqn='(Tins-Ta)*exp(-coeff_lambda*L/(Cp*Abs(m)))+Ta-Touts')
E2 = Eqn(name='E2', eqn='Tins+transpose(Vm)*Ts')
E3 = Eqn(name='E3', eqn='(Tinr-Ta)*exp(-coeff_lambda*L/(Cp*Abs(m)))+Ta-Toutr')
E4 = Eqn(name='E4', eqn='Tinr-transpose(Vp)*Tr')
E5 = Eqn(name='E5', eqn='V_rs*m+A_rs*mq')
E6 = Eqn(name='E6', eqn='V_l*m-A_l*mq')
E7 = Eqn(name='E7', eqn=m[i])
E8 = Eqn(name='E8', eqn=mL * K * Abs(m) * m)
E9 = Eqn(name='E9', eqn='Diagonal(A_li*Ts)*V_p_li*Abs(m)-V_p_li*Diagonal(Touts)*Abs(m)')
E10 = Eqn(name='E10', eqn='Diagonal(A_rsi*Tr)*V_m_rsi*Abs(m)-V_m_rsi*Diagonal(Toutr)*Abs(m)')
E11 = Eqn(name='E11', eqn='A_rsl*phi-4182*Diagonal(A_rsl*mq)*(A_rsl*Ts-A_rsl*Tr)')
E12 = Eqn(name='E12', eqn='A_rs*Ts-Ts_set')
E13 = Eqn(name='E13', eqn='A_l*Tr-Tr_set')
E14 = Eqn(name='E14', eqn='A_sl*phi-phi_set')
E15 = Eqn(name='E15', eqn='A_i*phi')
E16 = Eqn(name='E16', eqn=mq[i])
E = AE(name='Pipe Equations', eqn=[E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16])

y0 = Vars(list(var_dict.values()))

y_nr = nr_method(E, y0)
y_cnr = continuous_nr(E, y0)

sys_df = pd.read_excel('instances/4node3pipe_bench.xlsx',
                       sheet_name=None,
                       engine='openpyxl',
                       header=None
                       )


def test_nr_method():
    for var_name in ['Ts', 'Tr', 'm', 'mq', 'phi']:
        # find nonzero elements
        idx_nonzero = np.nonzero(y_nr[var_name])
        assert max(abs((y_nr[var_name][idx_nonzero] - np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, ))) /
                   np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, )) <= 1e-8


def test_cnr_method():
    for var_name in ['Ts', 'Tr', 'm', 'mq', 'phi']:
        # find nonzero elements
        idx_nonzero = np.nonzero(y_cnr[var_name])
        assert max(abs((y_cnr[var_name][idx_nonzero] - np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, )) /
                       np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, ))) <= 1e-8
