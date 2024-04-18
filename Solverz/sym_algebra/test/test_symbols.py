from Solverz.sym_algebra.symbols import iVar, Solverz_internal_name


def test_internal_var_declaration():

    for name in Solverz_internal_name:
        try:
            iVar(name)
        except ValueError as v:
            assert v.args[0] == f'Solverz built-in name {name}, cannot be used as variable name.'

