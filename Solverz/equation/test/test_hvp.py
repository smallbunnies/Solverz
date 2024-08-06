from numbers import Number

import numpy as np
from numpy.testing import assert_allclose
import re
import pytest

from sympy import Integer

from Solverz.equation.eqn import Eqn, Ode
from Solverz.equation.equations import DAE, AE
from Solverz.equation.param import Param
from Solverz.sym_algebra.symbols import iVar, idx, Para
from Solverz.sym_algebra.functions import exp, sin, cos, Diag
from Solverz.variable.variables import combine_Vars, as_Vars
from Solverz.equation.jac import JacBlock, Ones, Jac
from Solverz.equation.hvp import Hvp


def test_hvp():
    jac = Jac()
    x = iVar("x")
    jac.add_block(
        "a",
        x[0],
        JacBlock(
            "a",
            slice(0, 1),
            x[0],
            np.array([1]),
            slice(0, 2),
            exp(x[0]),
            np.array([2.71828183]),
        ),
    )
    jac.add_block(
        "a",
        x[1],
        JacBlock(
            "a",
            slice(0, 1),
            x[1],
            np.array([1]),
            slice(0, 2),
            cos(x[1]),
            np.array([0.54030231]),
        ),
    )
    jac.add_block(
        "b",
        x[0],
        JacBlock("b",
                 slice(1, 2),
                 x[0],
                 np.ones(1),
                 slice(0, 2),
                 1,
                 np.array([1])),
    )
    jac.add_block(
        "b",
        x[1],
        JacBlock("b", slice(1, 2), x[1], np.ones(1), slice(0, 2), 2 * x[1], np.array([2])),
    )

    h = Hvp(jac)
    v_ = Para("v_", internal_use=True)
    assert h.blocks_sorted['a'][x[0]].DeriExpr == v_[0]*exp(x[0])
    # ISSUE: the two symbolic expressions below cannot be equal, we believe this is a sympy issue.
    assert h.blocks_sorted['a'][x[1]].DeriExpr.__repr__() == (-v_[1]*sin(x[1])).__repr__()
    assert h.blocks_sorted['b'][x[1]].DeriExpr == 2*v_[1]
