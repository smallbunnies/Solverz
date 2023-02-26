import numpy as np
import pandas as pd
from numpy import linalg

from Solverz.eqn import Ode
from Solverz.equations import DAE
from Solverz.solver import ode45, nrtp45
from Solverz.var import TimeVar
from Solverz.variables import TimeVars

vdp1_1 = Ode(name='vdp1_1', e_str='x2', diff_var='x1')
vdp1_2 = Ode(name='vdp1_2', e_str='(1-x1**2)*x2-x1', diff_var='x2')
vdp1 = DAE([vdp1_1, vdp1_2], name='vdp1')
x1 = TimeVar('x1')
x1.v0 = [2]
x2 = TimeVar('x2')
x2.v0 = [0]

T = 10

t, y, kout = ode45(vdp1, TimeVars([x1, x2], length=100), [0, T], output_k=True)
tspan = np.linspace(0, 10, 1001)
yinterp = np.empty(shape=[2, 0])
for i in range(t.shape[0] - 1):
    yinterp = np.concatenate(
        [yinterp, nrtp45(tspan[np.where(((t[i] <= tspan) & (tspan <= t[i + 1])))], t[i], y.array[:, i],
                         t[i + 1] - t[i], kout[i])], axis=1)

df = pd.read_excel('../instances/test_ode45.xlsx',
                   sheet_name=None,
                   engine='openpyxl',
                   header=None
                   )


def test_ode45():
    assert linalg.norm(yinterp.T - df['Sheet1']) <= 1e-11
