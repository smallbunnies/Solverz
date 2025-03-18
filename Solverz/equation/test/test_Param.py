import numpy as np
from scipy.sparse import coo_array

from Solverz.equation.param import Param, IdxParam, TimeSeriesParam
from Solverz.utilities.testing import assert_allclose_sparse


def test_Param():
    # test of Param basic
    Ts1 = Param(name='Ts')
    try:
        Ts1.v = [[100, 100]]
    except ValueError as e:
        assert e.args[0] == 'Input list dim 2 higher than dim set to be 1'
    Ts1.v = [100, 100]
    assert Ts1.v.__str__() == '[100. 100.]'

    Ts2 = Param(name='Ts', value=[1, 2, 3], dim=1)
    assert Ts2.v.__str__() == '[1. 2. 3.]'

    Ts3 = Param(name='Ts', value=[1, 2, 3], dim=2)
    assert Ts3.v.__str__() == '[[1.]\n [2.]\n [3.]]'

    Ts4 = Param(name='Ts', value=[1, 2, 3], dim=2, sparse=True)
    assert_allclose_sparse(Ts4.v, coo_array(np.array([1, 2, 3]).reshape((-1, 1)).astype(float)))

    Ts5 = Param(name='Ts', value=[1, 2, 3], dim=2, sparse=False)
    assert Ts5.v.__str__() == '[[1.]\n [2.]\n [3.]]'

    v = np.array([[1, 0], [2, 9], [0, 3]])
    A = Param(name='A', value=v, dim=2, sparse=True)
    assert_allclose_sparse(A.v, coo_array(v))

    # test of IdxParam
    f = IdxParam(name='f',
                 value=[1, 2, 3])

    try:
        t = IdxParam(name='t',
                     value=[[1, 2, 3]])
    except ValueError as e:
        assert e.args[0] == 'Input list dim 2 higher than dim set to be 1'

    # test of TimeSeriesParam
    Pb = TimeSeriesParam(name='Pb',
                         v_series=[0, 10, 100],
                         time_series=[0, 10, 20],
                         value=0)
    assert Pb.get_v_t(5).__str__() == '[5.]'

    try:
        Pb = TimeSeriesParam(name='Pb',
                             v_series=[0, 10, 100],
                             time_series=[0, -10, 20],
                             value=0)
    except ValueError as e:
        assert e.args[0] == 'Time stamp should be strictly monotonically increasing!'

    try:
        Pb = TimeSeriesParam(name='Pb',
                             v_series=[0, 10, 100, 200],
                             time_series=[0, 10, 20],
                             value=0)
    except ValueError as e:
        assert e.args[0] == 'Incompatible length between value series and time series!'

    G = TimeSeriesParam(name='G',
                        v_series=[2, 10000, 10000, 2, 2],
                        time_series=[0, 0.002, 0.03, 0.032, 10],
                        value=np.array([[1, 0, 3], [0, 0.1, -0.4], [1.2, -np.pi, 0]]),
                        index=(1, 1),
                        dim=2,
                        sparse=True)
    assert (G.get_v_t(0.001).toarray().__str__() ==
            '[[ 1.00000000e+00  0.00000000e+00  3.00000000e+00]\n [ 0.00000000e+00  5.00100000e+03 -4.00000000e-01]\n [ 1.20000000e+00 -3.14159265e+00  0.00000000e+00]]')
