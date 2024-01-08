import numpy as np

from Solverz import Eqn, AE, nr_method, as_Vars, Var, made_numerical, parse_ae_v

x = Var(name='x', value=2)

e = Eqn(name='e', eqn=x ** 2 - 1)
f = AE(name='F',
       eqn=e)
y = as_Vars(x)
nf = made_numerical(f, y)
y1 = nr_method(nf, y.array)
y = parse_ae_v(y1, y.a)


def test_nr_method():
    assert abs(y['x'] - 1) <= 1e-8
