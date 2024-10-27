from ..functions import sin, cos
from Solverz import Var

def test_function_arg_parser():
    # The functions parse the Solverz.ssymbol as symbols.
    assert sin(Var('x')).args[0].name == 'x'
    assert cos(Var('x')).args[0].name == 'x'
