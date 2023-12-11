from Solverz.symboli_algebra.symbols import Var, Para, idx
from Solverz.symboli_algebra.functions import Abs, Mat_Mul
from Solverz.symboli_algebra.matrix_calculus import TensorExpr
from Solverz.equation.eqn import Eqn

mL = Para('mL', dim=2)
K = Para(name='K')
m = Var(name='m')
V = Para(name='V', dim=2)
Ts = Var(name='Ts')
Vp = Para(name='Vp', dim=2)
Touts = Var(name='Touts')
li = idx(name='li')
mq = Var(name='mq')
rs = idx(name='rs')
E8 = Eqn(name='E8', eqn=Mat_Mul(mL, (K * Abs(m) * m)))
E5 = Eqn(name='E5', eqn=Mat_Mul(V[rs, :], m) - mq[rs])
E9 = Eqn(name='E9', eqn=Ts[li] * Mat_Mul(Vp[li, :], Abs(m)) - Mat_Mul(Vp[li, :], (Touts * Abs(m))))
TE8 = TensorExpr(E8.RHS)
TE5 = TensorExpr(E5.RHS)
TE9 = TensorExpr(E9.RHS)

e = Var(name='e', value=[1.06, 1, 1.00])
f = Var(name='f', value=[0, 0, 0])
p = Var(name='p')
q = Var(name='q')
G = Para(name='G', dim=2, value=[0])
B = Para(name='B', dim=2, value=[0])

P1 = Eqn(name='Active power balance of PQ',
         eqn=e * (Mat_Mul(G, e) - Mat_Mul(B, f)) + f * (Mat_Mul(B, e) + Mat_Mul(G, f)) - p)
P2 = Eqn(name='Reactive power balance of PQ',
         eqn=f * (Mat_Mul(G, e) - Mat_Mul(B, f)) - e * (Mat_Mul(B, e) + Mat_Mul(G, f)) - q)
TP1 = TensorExpr(P1.RHS)
TP2 = TensorExpr(P2.RHS)


def test_matrix_calculus():
    assert TE8.diff(m).__repr__() == 'mL@(diag(K*Abs(m)) + diag(K*m*sign(m)))'
    assert TE5.diff(m).__repr__() == 'V[rs,::]'
    assert TE5.diff(mq[rs]).__repr__() == '-1'
    assert TE9.diff(m).__repr__() == '-Vp[li,::]@diag(Touts*sign(m)) + diag(Ts[li])@Vp[li,::]@diag(sign(m))'
    assert TE9.diff(Ts[li]).__repr__() == 'diag(Vp[li,::]@Abs(m))'
    assert TE9.diff(Touts).__repr__() == '-Vp[li,::]@diag(Abs(m))'
    assert TP1.diff(e).__repr__() == 'diag(-B@f + G@e) + diag(e)@G + diag(f)@B'
    assert TP1.diff(f).__repr__() == 'diag(B@e + G@f) + diag(e)@(-B) + diag(f)@G'
    assert TP2.diff(e).__repr__() == '-(diag(B@e + G@f) + diag(e)@B) + diag(f)@G'
    assert TP2.diff(f).__repr__() == 'diag(-B@f + G@e) - diag(e)@G + diag(f)@(-B)'
