from typing import Dict, Union
from sympy import Integer, Expr

from Solverz.sym_algebra.symbols import iVar, IdxVar, Para
from Solverz.equation.jac import Jac, JacBlock
from Solverz.utilities.address import Address
from Solverz.utilities.type_checker import is_zero

SolVar = Union[iVar, IdxVar]


class Hvp:

    def __init__(self, jac: Jac) -> None:
        self.blocks_sorted: Dict[str, Dict[SolVar, JacBlock]] = dict()
        self.jac0 = jac
        self.jac1 = Jac()

        # first multiply jac by vector v
        self.eqn_column: Dict[str, Expr] = dict()
        v = Para("v_", internal_use=True)
        for eqn_name, jbs_row in self.jac0.blocks.items():
            expr = Integer(0)
            for var, jb in jbs_row.items():
                den_var_addr = parse_den_var_addr(jb.DenVarAddr)
                match jb.DiffVarType:
                    case "scalar":
                        match jb.DeriType:
                            case "scalar":
                                expr += jb.DeriExpr * v[den_var_addr]
                            case "vector":
                                expr += jb.DeriExpr * v[den_var_addr]
                            case _:
                                raise TypeError(
                                    f"Unknown Derivative type {jb.DiffVarType}!"
                                )
                    case "vector":
                        match jb.DeriType:
                            case "scalar":
                                expr += jb.DeriExpr * v[den_var_addr]
                            case "vector":
                                expr += jb.DeriExpr * v[den_var_addr]
                            case "matrix":
                                raise NotImplementedError(
                                    "Matrix derivative not implemented in Hvp!"
                                )
                            case _:
                                raise TypeError(
                                    f"Unknown Derivative type {jb.DiffVarType}!"
                                )
                    case _:
                        raise TypeError(f"Unknown DiffVarType {jb.DiffVarType}!")
            self.eqn_column[eqn_name] = expr

        # derive Jac
        for eqn_name, jbs_row in self.jac0.blocks.items():
            for var, jb in jbs_row.items():
                DeriExpr = self.eqn_column[eqn_name].diff(jb.DiffVar)
                if not is_zero(DeriExpr):
                    self.jac1.add_block(
                        eqn_name,
                        var,
                        JacBlock(
                            eqn_name,
                            jb.EqnAddr,
                            jb.DiffVar,
                            jb.DiffVarValue,
                            jb.VarAddr,
                            DeriExpr,
                            jb.Value0,
                        ),
                    )

        self.blocks_sorted = self.jac1.blocks_sorted


def parse_den_var_addr(den_var_addr: slice | int):
    if isinstance(den_var_addr, int):
        den_var_addr = slice(den_var_addr, den_var_addr + 1)
    if den_var_addr.stop - den_var_addr.start == 1:
        return den_var_addr.start
    else:
        return den_var_addr
