import pandas as pd
import numpy as np
from functools import partial

from Solverz.eqn import Eqn
from Solverz.equations import AE
from Solverz.solver import nr_method
from Solverz.variables import Vars
from Solverz.var import Var
from Solverz.param import Param


# %% initialize variables and params
sys_df = pd.read_excel('instances/4node3pipe_change_sign.xlsx',
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
param_dict['A_i'] = Param(name='A_i', value=np.asarray(sys_df['A_i']))
param_dict['m_L'] = Param(name='m_L', value=np.asarray(sys_df['m_L']))
param_dict['K'] = Param(name='K', value=np.asarray(sys_df['K']).reshape(-1, ))
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
E1 = Eqn(name='E1',
         e_str='(Tins-Ta)*exp(-coeff_lambda*L/(Cp*Abs(m)))+Ta-Touts',
         commutative=True)

E2 = Eqn(name='E2',
         e_str='Tins+transpose(Vm)*Ts',
         commutative=False)

E3 = Eqn(name='E3',
         e_str='(Tinr-Ta)*exp(-coeff_lambda*L/(Cp*Abs(m)))+Ta-Toutr',
         commutative=True)

E4 = Eqn(name='E4',
         e_str='Tinr-transpose(Vp)*Tr',
         commutative=False)

E5 = Eqn(name='E5',
         e_str='V_rs*m+A_rs*mq',
         commutative=False)

E6 = Eqn(name='E6',
         e_str='V_l*m-A_l*mq',
         commutative=False)

E7 = Eqn(name='E7',
         e_str='V_i*m',
         commutative=False)

E8 = Eqn(name='E8',
         e_str='m_L*Diagonal(K)*Diagonal(Abs(m))*m',
         commutative=False)

E9 = Eqn(name='E9',
         e_str='Diagonal(A_li*Ts)*V_p_li*Abs(m)-V_p_li*Diagonal(Touts)*Abs(m)',
         commutative=False)

E10 = Eqn(name='E10',
          e_str='Diagonal(A_rsi*Tr)*V_m_rsi*Abs(m)-V_m_rsi*Diagonal(Toutr)*Abs(m)',
          commutative=False)

E11 = Eqn(name='E11',
          e_str='A_rsl*phi-4182*Diagonal(A_rsl*mq)*(A_rsl*Ts-A_rsl*Tr)',
          commutative=False)

E12 = Eqn(name='E12',
          e_str='A_rs*Ts-Ts_set',
          commutative=False)

E13 = Eqn(name='E13',
          e_str='A_l*Tr-Tr_set',
          commutative=False)

E14 = Eqn(name='E14',
          e_str='A_sl*phi-phi_set',
          commutative=False)

E15 = Eqn(name='E15',
          e_str='A_i*phi',
          commutative=False)

E16 = Eqn(name='E16',
          e_str='A_i*mq',
          commutative=False)

AE1 = AE(name='Pipe Equations',
         eqn=[E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16],
         param=list(param_dict.values()))

y0 = Vars(list(var_dict.values()))
# %%
y_nr = nr_method(AE1, y0)

# y_cnr = continuous_nr(deepcopy(E), deepcopy(y0))

sys_df = pd.read_excel('instances/4node3pipe_change_sign_bench.xlsx',
                       sheet_name=None,
                       engine='openpyxl',
                       header=None
                       )


def test_nr_method():
    for var_name in ['Ts', 'Tr', 'm', 'mq', 'phi']:
        # find nonzero elements
        idx_nonzero = np.nonzero(y_nr[var_name])
        assert max(abs((y_nr[var_name][idx_nonzero] - np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, )) /
                       np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1, ))) <= 1e-8

# def test_cnr_method():
#     for var_name in ['Ts', 'Tr', 'm', 'mq', 'phi']:
#         # find nonzero elements
#         idx_nonzero = np.nonzero(y_cnr[var_name].array)
#         assert max(abs((y_cnr[var_name][idx_nonzero] - np.asarray(sys_df[var_name])[idx_nonzero]) /
#                        np.asarray(sys_df[var_name])[idx_nonzero])) <= 1e-8
