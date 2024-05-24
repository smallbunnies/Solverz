import numpy as np


class Jac:
    pass


class JacBlock:

    def __init__(self,
                 eqn_name,
                 eqn_addr,
                 var_name,
                 var_addr,
                 expr,
                 value0):
        self.eqn_name = eqn_name
        self.eqn_addr = eqn_addr
        self.var_name = var_name
        self.var_addr = var_addr
        self.expr = expr
        self.value0 = value0


def inline_printer():
    pass


def module_printer():
    pass
