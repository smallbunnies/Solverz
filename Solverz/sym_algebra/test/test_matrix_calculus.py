from Solverz.sym_algebra.symbols import iVar, Para, idx
from Solverz.sym_algebra.functions import Abs, Mat_Mul, Diag, transpose, exp, sin, cos, ln
from Solverz.sym_algebra.matrix_calculus import TensorExpr, MixedEquationDiff
from Solverz.equation.eqn import Eqn

mL = Para('mL', dim=2)
K = Para(name='K')
m = iVar(name='m')
V = Para(name='V', dim=2)
Ts = iVar(name='Ts')
Vp = Para(name='Vp', dim=2)
Touts = iVar(name='Touts')
li = idx(name='li')
mq = iVar(name='mq')
rs = idx(name='rs')
E8 = Eqn(name='E8', eqn=Mat_Mul(mL, (K * Abs(m) * m)))
E5 = Eqn(name='E5', eqn=Mat_Mul(V[rs, :], m) - mq[rs])
E9 = Eqn(name='E9', eqn=Ts[li] * Mat_Mul(Vp[li, :], Abs(m)) - Mat_Mul(Vp[li, :], (Touts * Abs(m))))
TE8 = TensorExpr(E8.RHS)
TE5 = TensorExpr(E5.RHS)
TE9 = TensorExpr(E9.RHS)

e = iVar(name='e', value=[1.06, 1, 1.00])
f = iVar(name='f', value=[0, 0, 0])
P = iVar(name='P')
q = iVar(name='q')
G = Para(name='G', dim=2, value=[0])
B = Para(name='B', dim=2, value=[0])

P1 = Eqn(name='Active power balance of PQ',
         eqn=e * (Mat_Mul(G, e) - Mat_Mul(B, f)) + f * (Mat_Mul(B, e) + Mat_Mul(G, f)) - P)
P2 = Eqn(name='Reactive power balance of PQ',
         eqn=f * (Mat_Mul(G, e) - Mat_Mul(B, f)) - e * (Mat_Mul(B, e) + Mat_Mul(G, f)) - q)
TP1 = TensorExpr(P1.RHS)
TP2 = TensorExpr(P2.RHS)


def test_matrix_calculus():
    assert TE8.diff(m).__repr__() == 'mL@(diag(K*Abs(m)) + diag(K*m*Sign(m)))'
    assert TE5.diff(m).__repr__() == 'V[rs,:]'
    assert TE5.diff(mq[rs]).__repr__() == '-1'
    assert TE9.diff(m).__repr__() == '-Vp[li,:]@diag(Touts*Sign(m)) + diag(Ts[li])@Vp[li,:]@diag(Sign(m))'
    assert TE9.diff(Ts[li]).__repr__() == 'diag(Vp[li,:]@Abs(m))'
    assert TE9.diff(Touts).__repr__() == '-Vp[li,:]@diag(Abs(m))'
    assert TP1.diff(e).__repr__() == 'diag(-B@f + G@e) + diag(e)@G + diag(f)@B'
    assert TP1.diff(f).__repr__() == 'diag(B@e + G@f) + diag(e)@(-B) + diag(f)@G'
    assert TP2.diff(e).__repr__() == '-(diag(B@e + G@f) + diag(e)@B) + diag(f)@G'
    assert TP2.diff(f).__repr__() == 'diag(-B@f + G@e) - diag(e)@G + diag(f)@(-B)'


# --- New tests for extended operations ---

def test_exp():
    """d/dx exp(A@x) = diag(exp(A@x)) @ A"""
    Am = Para('Am', dim=2)
    xv = iVar('xv')
    expr = exp(Mat_Mul(Am, xv))
    te = TensorExpr(expr)
    assert te.diff(xv).__repr__() == 'diag(exp(Am@xv))@Am'


def test_sin():
    """d/dx sin(A@x) = diag(cos(A@x)) @ A"""
    Am = Para('Am', dim=2)
    xv = iVar('xv')
    expr = sin(Mat_Mul(Am, xv))
    te = TensorExpr(expr)
    assert te.diff(xv).__repr__() == 'diag(cos(Am@xv))@Am'


def test_ln():
    """d/dx ln(A@x) = diag((A@x)^{-1}) @ A"""
    Am = Para('Am', dim=2)
    xv = iVar('xv')
    expr = ln(Mat_Mul(Am, xv))
    te = TensorExpr(expr)
    assert te.diff(xv).__repr__() == 'diag(Am@xv**(-1))@Am'


def test_pow():
    """d/dx A@(x**2) = A @ diag(2*x)"""
    Am = Para('Am', dim=2)
    xv = iVar('xv')
    expr = Mat_Mul(Am, xv ** 2)
    te = TensorExpr(expr)
    assert te.diff(xv).__repr__() == 'Am@diag(2*xv)'


def test_transpose():
    """d/dx transpose(A)@x = transpose(A)"""
    Am = Para('Am', dim=2)
    xv = iVar('xv')
    expr = Mat_Mul(transpose(Am), xv)
    te = TensorExpr(expr)
    assert te.diff(xv).__repr__() == 'transpose(Am)'


def test_diag_input():
    """d/dx Diag(x)@A@y = diag(A@y), d/dy = diag(x)@A"""
    Am = Para('Am', dim=2)
    xv = iVar('xv')
    yv = iVar('yv')
    expr = Mat_Mul(Diag(xv), Mat_Mul(Am, yv))
    te = TensorExpr(expr)
    assert te.diff(xv).__repr__() == 'diag(Am@yv)'
    assert te.diff(yv).__repr__() == 'diag(xv)@Am'


def test_combination_mul():
    """d/dx sin(A@x) * B@y = diag(B@y) @ diag(cos(A@x)) @ A"""
    Am = Para('Am', dim=2)
    Bm = Para('Bm', dim=2)
    xv = iVar('xv')
    yv = iVar('yv')
    expr = sin(Mat_Mul(Am, xv)) * Mat_Mul(Bm, yv)
    te = TensorExpr(expr)
    assert te.diff(xv).__repr__() == 'diag(Bm@yv)@diag(cos(Am@xv))@Am'
    assert te.diff(yv).__repr__() == 'diag(sin(Am@xv))@Bm'


def test_combination_add():
    """d/dx (exp(A@x) + B@y) = diag(exp(A@x))@A, d/dy = B"""
    Am = Para('Am', dim=2)
    Bm = Para('Bm', dim=2)
    xv = iVar('xv')
    yv = iVar('yv')
    expr = exp(Mat_Mul(Am, xv)) + Mat_Mul(Bm, yv)
    te = TensorExpr(expr)
    assert te.diff(xv).__repr__() == 'diag(exp(Am@xv))@Am'
    assert te.diff(yv).__repr__() == 'Bm'


def test_mixed_equation_diff_api():
    """MixedEquationDiff API works correctly"""
    Am = Para('Am', dim=2)
    xv = iVar('xv')
    expr = Mat_Mul(Am, xv)
    result = MixedEquationDiff(expr, xv)
    assert result.__repr__() == 'Am'
