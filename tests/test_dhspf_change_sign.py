import pandas as pd

from Solverz.eqn import Eqn
from Solverz.equations import AE
from Solverz.miscellaneous import derive_dhs_param_var
from Solverz.solver import *
from Solverz.variables import Vars

var_dict, param_dict = derive_dhs_param_var('../instances/4node3pipe_change_sign.xlsx')

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

E = AE(name='Pipe Equations',
              eqn=[E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16],
              param=list(param_dict.values()))

y0 = Vars(list(var_dict.values()))

y_nr = nr_method(E, y0)

# y_cnr = continuous_nr(deepcopy(E), deepcopy(y0))

sys_df = pd.read_excel('../instances/4node3pipe_change_sign_bench.xlsx',
                       sheet_name=None,
                       engine='openpyxl',
                       header=None
                       )


def test_nr_method():
    for var_name in ['Ts', 'Tr', 'm', 'mq', 'phi']:
        # find nonzero elements
        idx_nonzero = np.nonzero(y_nr[var_name])
        assert max(abs((y_nr[var_name][idx_nonzero] - np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1,)) /
                       np.asarray(sys_df[var_name])[idx_nonzero].reshape(-1,))) <= 1e-8


# def test_cnr_method():
#     for var_name in ['Ts', 'Tr', 'm', 'mq', 'phi']:
#         # find nonzero elements
#         idx_nonzero = np.nonzero(y_cnr[var_name].array)
#         assert max(abs((y_cnr[var_name][idx_nonzero] - np.asarray(sys_df[var_name])[idx_nonzero]) /
#                        np.asarray(sys_df[var_name])[idx_nonzero])) <= 1e-8

