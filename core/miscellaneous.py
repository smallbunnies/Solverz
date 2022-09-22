import numpy as np
import pandas as pd

from .solverz_array import SolverzArray
from .var import Var
from .param import Param


def derive_incidence_matrix(f_node: np.ndarray,
                            t_node: np.ndarray):
    num_node = np.max([np.max(f_node), np.max(t_node)])
    num_pipe = f_node.shape[0]
    temp = np.zeros((num_node, num_pipe))
    for pipe in range(1, num_pipe + 1):
        temp[f_node[pipe - 1] - 1, pipe - 1] = -1
        temp[t_node[pipe - 1] - 1, pipe - 1] = 1

    return SolverzArray(temp)


def derive_v(v: np.ndarray,
             *args: np.ndarray):
    idx = np.array([], dtype=int)
    for arg in args:
        idx = np.append(idx, arg - 1)
    return v[idx, :].copy()


def derive_a(f_node: np.ndarray,
             t_node: np.ndarray,
             *args: np.ndarray):
    num_node = np.max([np.max(f_node), np.max(t_node)])
    temp = np.identity(num_node)
    idx = np.array([], dtype=int)
    for arg in args:
        idx = np.append(idx, arg - 1)
    return temp[idx, :].copy()


def derive_v_plus(v: SolverzArray):
    return (v + np.abs(v)) / 2


def derive_v_minus(v: SolverzArray):
    return (v - np.abs(v)) / 2


def derive_dhs_param_var(file: str):
    sys_df = pd.read_excel(file,
                           sheet_name=None,
                           engine='openpyxl',
                           )

    Cp = Param(name='Cp')
    Cp.v = np.array(sys_df['StaticHeatPipe'].Cp)
    L = Param(name='L')
    L.v = np.array(sys_df['StaticHeatPipe'].L)
    Ta = Param(name='Ta')
    Ta.v = np.array(sys_df['StaticHeatPipe'].Ta)
    coeff_lambda = Param(name='coeff_lambda')
    coeff_lambda.v = np.array(sys_df['StaticHeatPipe'].coeff_lambda)

    f_node = np.array(sys_df['StaticHeatPipe'].f_node)
    t_node = np.array(sys_df['StaticHeatPipe'].t_node)

    r_f_node = np.array(sys_df['Slack']['f_node'])
    s_f_node = np.array(sys_df['Source']['f_node'])
    l_f_node = np.array(sys_df['Load']['f_node'])
    i_f_node = np.array(sys_df['Intermediate']['f_node'])

    V = Param(name='V', info='Incidence Matrix')
    V.v = derive_incidence_matrix(f_node, t_node)
    Vp = Param(name='Vp', info='Positive part of incidence matrix')
    Vp.v = derive_v_plus(V.v)
    Vm = Param(name='Vm', info='Negative part of incidence matrix')
    Vm.v = derive_v_minus(V.v)

    # TODO: 声明变量、数组时必须强制声明变量大小 检查变量大小/参数大小与输入的数组大小是否匹配！

    Touts = Var(name='Touts')
    Touts.v = np.array(sys_df['StaticHeatPipe'].Touts)
    Toutr = Var(name='Toutr')
    Toutr.v = np.array(sys_df['StaticHeatPipe'].Toutr)
    Tins = Var(name='Tins')
    Tins.v = np.array(sys_df['StaticHeatPipe'].Tins)
    Tinr = Var(name='Tinr')
    Tinr.v = np.array(sys_df['StaticHeatPipe'].Tinr)
    Ts = Var(name='Ts')
    Ts.v = np.array(sys_df['Node'].Ts)
    Tr = Var(name='Tr')
    Tr.v = np.array(sys_df['Node'].Tr)
    m = Var(name='m')
    m.v = np.array(sys_df['StaticHeatPipe'].m)
    mq = Var(name='mq')
    mq.v = np.array(sys_df['Node'].mq)
    phi = Var(name='phi')
    phi.v = np.array(sys_df['Node'].phi)

    L = Param(name='L',
              value=np.transpose(SolverzArray(sys_df['Loop'].loc[0].values.tolist()[2:])))

    K = Param(name='K',
              value=SolverzArray(sys_df['StaticHeatPipe'].K.values))

    V_rs = Param(name='V_rs',
                 value=derive_v(np.asarray(V.v), r_f_node, s_f_node))

    V_l = Param(name='V_l',
                value=derive_v(np.asarray(V.v), l_f_node))

    V_i = Param(name='V_i',
                value=derive_v(np.asarray(V.v), i_f_node))

    V_p_li = Param(name='V_p_li',
                   value=derive_v(np.asarray(Vp.v), l_f_node, i_f_node))

    V_m_rsi = Param(name='V_m_rsi',
                    value=derive_v(np.asarray(Vm.v), r_f_node, s_f_node, i_f_node))

    A_rsl = Param(name='A_rsl',
                  value=derive_a(f_node,
                                 t_node,
                                 r_f_node,
                                 s_f_node,
                                 l_f_node))

    A_rs = Param(name='A_rs',
                 value=derive_a(f_node,
                                t_node,
                                r_f_node,
                                s_f_node))

    A_sl = Param(name='A_sl',
                 value=derive_a(f_node,
                                t_node,
                                s_f_node,
                                l_f_node))
    A_l = Param(name='A_l',
                value=derive_a(f_node,
                               t_node,
                               l_f_node))

    A_i = Param(name='A_i',
                value=derive_a(f_node,
                               t_node,
                               i_f_node))

    A_li = Param(name='A_li',
                 value=derive_a(f_node,
                                t_node,
                                l_f_node,
                                i_f_node))

    A_rsi = Param(name='A_rsi',
                  value=derive_a(f_node,
                                 t_node,
                                 r_f_node,
                                 s_f_node,
                                 i_f_node))

    Ts_set = Param(name='Ts_set',
                   value=[100])

    Tr_set = Param(name='Tr_set',
                   value=[50])

    phi_set = Param(name='phi_set',
                    value=np.array(sys_df['Node']['phi'][np.append(s_f_node, l_f_node)-1]))

    var_dict = {}
    param_dict = {}

    for param in [Cp, L, coeff_lambda, Ta, Vm, V_m_rsi, Vp, V_p_li, V_rs, A_li, A_rsi, V_i, V_l, A_rs, A_l, A_rsl, K, A_sl, A_i, Ts_set, Tr_set, phi_set]:
        param_dict[param.name] = param
    for var in [Ts, Tr, Tins, Tinr, Touts, Toutr, m, mq, phi]:
        var_dict[var.name] = var
    return var_dict, param_dict
