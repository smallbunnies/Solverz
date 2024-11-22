"""
Test the user defined modules.
"""


def test_udm():

    from Solverz import Model, Var, Eqn, made_numerical, MulVarFunc
    import numpy as np

    class Min(MulVarFunc):
        arglength = 2

        def fdiff(self, argindex=1):
            if argindex == 1:
                return dMindx(*self.args)
            elif argindex == 2:
                return dMindy(*self.args)

        def _numpycode(self, printer, **kwargs):
            return (f'myfunc.Min' + r'(' +
                    ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')')

    class dMindx(MulVarFunc):
        arglength = 2

        def _numpycode(self, printer, **kwargs):
            return (f'myfunc.dMindx' + r'(' +
                    ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')')

    class dMindy(MulVarFunc):
        arglength = 2

        def _numpycode(self, printer, **kwargs):
            return (f'myfunc.dMindy' + r'(' +
                    ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')')

    m = Model()
    m.x = Var('x', [1, 2])
    m.y = Var('y', [3, 4])
    m.f = Eqn('f', Min(m.x, m.y))
    sae, y0 = m.create_instance()
    ae = made_numerical(sae, y0, sparse=True)
    np.testing.assert_allclose(ae.F(y0, ae.p), np.array([1.0, 2.0]))
    np.testing.assert_allclose(ae.J(y0, ae.p).toarray(), np.array([[1., 0., 0., 0.],
                                                                   [0., 1., 0., 0.]]))

