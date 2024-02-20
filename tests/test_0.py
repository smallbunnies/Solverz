import numpy as np

from Solverz import Eqn, AE, nr_method, as_Vars, Var, made_numerical

x = Var(name='x', value=2)

e = Eqn(name='e', eqn=x ** 2 - 1)
f = AE(name='F',
       eqn=e)
y = as_Vars(x)
nf = made_numerical(f, y)
y = nr_method(nf, y)


def test_nr_method():
    assert abs(y['x'] - 1) <= 1e-8
