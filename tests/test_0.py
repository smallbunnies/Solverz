from core.eqn import Eqn
from core.equations import AE
from core.solver import nr_method
from core.var import Var
from core.variables import Vars

e = Eqn(name='e',
        e_str='X**2-1')
f = AE(name='F',
       eqn=e)
x = Var(name='X')
x.v = [2]
x = Vars([x])
x = nr_method(f, x)


def test_nr_method():
    assert abs(x['X'] - 1) <= 1e-8
