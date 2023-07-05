from Solverz.eqn import Eqn
from Solverz.param import Param

Apq = Param(name='Apq')
Gpq = Param(name='Gpq')
Bpq = Param(name='Bpq')

E1 = Eqn(name='Active power', eqn='-Apq*p+Diagonal(Apq*e)*(Gpq*e-Bpq*f)', commutative=False)

# E = Equations(E1,
#               name='AC Power flow',
#               param=[Apq, Bpq, Gpq])


