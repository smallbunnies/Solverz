from Solverz import Eqn, AE, nr_method, Var, Vars, Var_

X = Var_(name='X')

e = Eqn(name='e', eqn=X**2-1)
f = AE(name='F',
       eqn=e)
x = Var(name='X')
x.v = [2]
x = Vars([x])
x = nr_method(f, x)


def test_nr_method():
    assert abs(x['X'] - 1) <= 1e-8
