from Solverz.symboli_algebra.symbols import Var, Solverz_internal_name


def test_internal_var_declaration():

    for name in Solverz_internal_name:
        try:
            Var(name)
        except ValueError as v:
            assert v.args[0] == f'Solverz built-in name {name}, cannot be used as variable name.'

