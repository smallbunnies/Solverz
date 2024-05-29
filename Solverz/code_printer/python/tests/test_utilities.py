import pytest

from sympy import symbols, pycode
from sympy.codegen.ast import FunctionCall as SpFuncCall
from Solverz.code_printer.python.utilities import FunctionCall

expected = """
            The `args` parameter passed to sympy.codegen.ast.FunctionCall should not contain str, which may cause sympy parsing error. 
            For example, the sympy.codegen.ast.FunctionCall parses str E in args to math.e!
            """


def test_FunctionCall():
    E = symbols('E')
    assert pycode(FunctionCall('a', [E])) == 'a(E)'
    assert pycode(SpFuncCall('a', ['E'])) == 'a(math.e)'

    with pytest.raises(ValueError, match=expected):
        assert pycode(FunctionCall('a', ['E'])) == 'a(E)'
