from param import Param
import pandas as pd
import numpy as np
from eqn import Eqn, Equations
from var import Var, Vars
from miscellaneous import *
from routine import Routine
from solver import *

sys_df = pd.read_excel('4node3pipe.xlsx',
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

V = Param(name='V', info='Incidence Matrix')
V.v = derive_incidence_matrix(np.array(sys_df['StaticHeatPipe'].f_node), np.array(sys_df['StaticHeatPipe'].t_node))
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

E1 = Eqn(name='E1',
         e_str='(Tins-Ta)*exp(-coeff_lambda*L/(Cp*Abs(m)))+Ta-Touts',
         commutative=True)
# var=[Ts, Tr, Tins, Tinr, Touts, Toutr, m],
E2 = Eqn(name='E2',
         e_str='Tins+transpose(Vm)*Ts',
         commutative=False)
# var=[Ts, Tins],
E3 = Eqn(name='E3',
         e_str='(Tinr-Ta)*exp(-coeff_lambda*L/(Cp*Abs(m)))+Ta-Toutr',
         commutative=True)
# var=[Ts, Tr, Tins, Tinr, Touts, Toutr, m],
E4 = Eqn(name='E4',
         e_str='Tinr-transpose(Vp)*Tr',
         commutative=False)

# E5 = Eqn(name='E5',
#          e_str='V_rs*m+A_rs*mq',
#          commutative=False)
#
# E6 = Eqn(name='E6',
#          e_str='V_l*m-A_l*mq',
#          commutative=False)
#
# E7 = Eqn(name='E7',
#          e_str='V_i*m',
#          commutative=False)
#
# E8 = Eqn(name='E8',
#          e_str='L*Diagonal(m)*Abs(m)*m',
#          commutative=False)
#
# E9 = Eqn(name='E9',
#          e_str='Diagonal(Ts)*V_p*Abs(m)-V_p*Diagonal(Touts)*Abs(m)',
#          commutative=False)
#
# E10 = Eqn(name='E10',
#           e_str='Diagonal(Tr)*V_m*Abs(m)-V_m*Diagonal(Toutr)*Abs(m)',
#           commutative=False)
#
# E11 = Eqn(name='E11',
#           e_str='A_rsl*phi-4182*Diagonal(A_rsl*mq)*(A_rsl*Ts-A_rsl*Tr)',
#           commutative=False)

E = Equations(name='Pipe Equations',
              eqn=[E1, E2, E3, E4],
              param=[Cp, L, coeff_lambda, Ta, Vm, Vp])

y = Vars([Ts, Tr, Tins, Tinr, Touts, Toutr, m])

A = E.g_y(y)

DHSpf = Routine(E, y, tol=1e-8, solver=nr_method)

e = Eqn(name='e',
        e_str='x**2-1')
f = Equations(name='f',
              eqn=e)
x = Var(name='x')
x.v = [2]
x = Vars([x])
nr_method(f, x)
