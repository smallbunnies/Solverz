from Solverz import Eqn, AE, nr_method, Model, Var, made_numerical, Opt

m = Model()

m.x = Var(name='x', value=2)

m.e = Eqn(name='e', eqn=m.x ** 2 - 1)
m.F = AE(name='F', eqn=m.e)

g, y0 = m.create_instance()

nae = made_numerical(g, y0)
sol = nr_method(nae, y0, Opt(ite_tol=1e-8))


def test_nr_method():
    assert abs(sol.y['x'] - 1) <= 1e-8
