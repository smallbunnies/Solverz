import sympy as sp
import numpy as np
from core.var import *
from core.variables import TimeVars
from core.equations import AE, DAE
from core.eqn import Eqn, Ode, Pde
from core.param import Param
from core.solver import implicit_trapezoid
import matplotlib.pyplot as plt
import pandas as pd


x = TimeVar('x')
x.v = [0]
f1 = Ode(name='f', e_str='-2*(x-cos(t))', diff_var='x')

x = TimeVars(x, length=201)
x = implicit_trapezoid(DAE(f1), x, 0.1, 20)

plt.plot(np.arange(0, 20.1, 0.1), x.array.reshape((-1,)))

df = pd.read_excel('../instances/ode_test.xlsx',
                   sheet_name=None,
                   engine='openpyxl',
                   header=None
                   )


def test_ode():
    assert max(abs((x.array - np.asarray(df['Sheet1']).reshape(1, -1))).reshape(-1, )) <= 0.0009448126743213381
