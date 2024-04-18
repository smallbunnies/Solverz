from Solverz import Eqn, AE, nr_method, as_Vars, iVar, made_numerical

x = iVar(name='x', value=2)

e = Eqn(name='e', eqn=x ** 2 - 1)
f = AE(name='F',
       eqn=e)
y = as_Vars(x)
nf = made_numerical(f, y)
sol = nr_method(nf, y)


def test_nr_method():
    assert abs(sol.y['x'] - 1) <= 1e-8
