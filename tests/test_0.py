from Solverz.eqn import Eqn
from Solverz.equations import AE
from Solverz.solvers.aesolver import nr_method
from Solverz.var import Var
from Solverz.variables import Vars

e = Eqn(name='e', eqn='X**2-1')
f = AE(name='F',
       eqn=e)
x = Var(name='X')
x.v = [2]
x = Vars([x])
x = nr_method(f, x)


def test_nr_method():
    assert abs(x['X'] - 1) <= 1e-8
