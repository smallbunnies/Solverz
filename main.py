from param import Param
import pandas as pd
import numpy as np
from eqn import Eqn
from var import Var
from miscellaneous import *

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
Touts.v = sys_df['StaticHeatPipe'].Touts
Toutr = Var(name='Toutr')
Toutr.v = sys_df['StaticHeatPipe'].Toutr
Tins = Var(name='Tins')
Tins.v = sys_df['StaticHeatPipe'].Tins
Tinr = Var(name='Tinr')
Tinr.v = sys_df['StaticHeatPipe'].Tinr
Ts = Var(name='Ts')
Ts.v = sys_df['Node'].Ts
Tr = Var(name='Tr')
Tr.v = sys_df['Node'].Tr
m = Var(name='m')
m.v = sys_df['StaticHeatPipe'].m

EqnPipe1 = Eqn(name='PipeEqn1',
               e_str='(Tins-Ta)*exp(-coeff_lambda*L/(Cp*Abs(m)))+Ta-Touts',
               var=[Ts, Tr, Tins, Tinr, Touts, Toutr, m],
               param=[Cp, L, coeff_lambda, Ta],
               commutative= True)
EqnPipe2 = Eqn(name='PipeEqn2',
               e_str='Tins+Transposes(Vm)*Ts',
               var=[Ts, Tins],
               param=Vm,
               commutative=False)
EqnPipe3 = Eqn(name='PipeEqn3',
               e_str='(Tinr-Ta)*exp(-coeff_lambda*L/(Cp*Abs(m)))+Ta-Toutr',
               var=[Ts, Tr, Tins, Tinr, Touts, Toutr, m],
               param=[Cp, L, coeff_lambda, Ta],
               commutative=True)
EqnPipe4 = Eqn(name='PipeEqn4',
               e_str='Tinr-Transposes(Vp)*Tr',
               var=[Tr, Tinr],
               param=Vp,
               commutative=False)
# FIXME: PipeEqn1 is not commutative why?
EqnPipe = Eqn(name=['PipeEqn1', 'PipeEqn2', 'PipeEqn3', 'PipeEqn4'],
              e_str=['(Tins-Ta)*exp(-coeff_lambda*L/(Cp*Abs(m)))+Ta-Touts',
                     'Tins+Transposes(Vm)*Ts',
                     '(Tinr-Ta)*exp(-coeff_lambda*L/(Cp*Abs(m)))+Ta-Toutr',
                     'Tinr-Transposes(Vp)*Tr'],
              var=[Ts, Tr, Tins, Tinr, Touts, Toutr, m],
              param=[Cp, L, coeff_lambda, Ta, Vm, Vp],
              commutative=[True, False, True, False])
