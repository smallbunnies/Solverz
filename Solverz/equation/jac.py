from __future__ import annotations
import numpy as np
from typing import Dict, Union, List
import warnings

from sympy import Expr, Function, Integer
from Solverz.sym_algebra.symbols import iVar, IdxVar
from Solverz.utilities.type_checker import is_vector, is_scalar, is_integer, is_number, PyNumber, is_zero
from Solverz.sym_algebra.functions import Diag, Ones


SolVar = Union[iVar, IdxVar]


class Jac:

    def __init__(self):
        self.blocks: Dict[str, Dict[SolVar, JacBlock]] = dict()
        self.is_sorted = False
        self.__blocks_sorted: Dict[str, Dict[SolVar, JacBlock]] = dict()

    def add_block(self,
                  eqn_name: str,
                  diff_var: SolVar,
                  jb: JacBlock):
        if eqn_name in self.blocks:
            self.blocks[eqn_name][diff_var] = jb
        else:
            self.blocks[eqn_name] = dict()
            self.blocks[eqn_name][diff_var] = jb
        self.is_sorted = False

    def sort_blocks(self):
        sorted_blocks = dict()
        for eqn_name in sorted(self.blocks):
            dict_var_jb = self.blocks[eqn_name]
            # SolVar cannot be directly used in sorted, so translate it to string first, then recover
            dict_varname_var = {var.__repr__(): var for var in dict_var_jb.keys()}
            sorted_blocks[eqn_name] = {dict_varname_var[varname]: dict_var_jb[dict_varname_var[varname]] for varname in
                                       sorted(dict_varname_var)}
        self.__blocks_sorted = sorted_blocks
        self.is_sorted = True

    @property
    def blocks_sorted(self) -> Dict[str, Dict[SolVar, JacBlock]]:
        if self.is_sorted:
            return self.__blocks_sorted
        else:
            self.sort_blocks()
            return self.__blocks_sorted

    @property
    def JacEleNum(self) -> int:
        num = 0
        for jbs_row in self.blocks.values():
            for jb in jbs_row.values():
                if jb.DeriType == 'matrix':
                    raise NotImplementedError("Matrix derivative type not supported!")
                else:
                    num = num + jb.SpEleSize
        return num

    def parse_row_col_data(self):
        """
        Parse the row, col and data for sparse coo-jac construction.
        """
        if not self.is_sorted:
            self.sort_blocks()
        row = np.zeros(self.JacEleNum, dtype=int)
        col = np.zeros(self.JacEleNum, dtype=int)
        data = np.zeros(self.JacEleNum, dtype=float)
        addr_by_ele_0 = 0
        for eqn_name, jbs_row in self.blocks_sorted.items():
            for var, jb in jbs_row.items():
                addr_by_ele = slice(addr_by_ele_0, addr_by_ele_0 + jb.SpEleSize)
                row[addr_by_ele] = jb.SpEqnAddr.copy()
                col[addr_by_ele] = jb.SpVarAddr.copy()
                if jb.IsDeriNumber:
                    data[addr_by_ele] = jb.DeriExpr
                addr_by_ele_0 += jb.SpEleSize
        return row, col, data


class JacBlock:

    def __init__(self,
                 EqnName: str,
                 EqnAddr: slice,
                 DiffVar,
                 DiffVarValue,
                 VarAddr,
                 DeriExpr: Expr,
                 Value0: np.ndarray | PyNumber):
        """
        var_addr is the address of the non-indexed variable. Fox example, if the diff_var is x[0], then the
        var_addr is the address of x. JacBlock will parse the address of x[0]

        jac type:
        0. a column vector
        1. a diagonal matrix
        2. a matrix
        """
        if is_zero(DeriExpr):
            raise ValueError(f"We wont allow {DeriExpr} derivative!")

        self.EqnName = EqnName
        self.EqnAddr: slice = EqnAddr
        self.DiffVar: SolVar = DiffVar
        self.DiffVarValue: np.ndarray = DiffVarValue
        self.VarAddr: slice = VarAddr
        self.DeriExpr = DeriExpr
        self.DeriExprBc = Integer(0)  # Derivative broadcast
        self.Value0 = Value0  # the value of derivative
        self.SpEqnAddr: np.ndarray = np.array([])
        self.SpVarAddr: np.ndarray = np.array([])
        self.SpEleSize = 0
        self.SpDeriExpr: Expr = Integer(0)
        self.DenEqnAddr: slice = slice(0)
        self.DenVarAddr: slice | int = slice(0)
        self.DenDeriExpr: Expr = Integer(0)

        EqnSize = self.EqnAddr.stop - self.EqnAddr.start

        # find out jac type
        if DiffVarValue.size > 1:
            DiffVarType = 'vector'
        else:
            DiffVarType = 'scalar'

        self.DiffVarType = DiffVarType

        if not isinstance(self.Value0, np.ndarray):
            if isinstance(self.Value0, PyNumber):
                self.Value0 = np.array(self.Value0)
            else:
                raise TypeError(f"Derivative value type {type(self.Value0)} not supported!")

        if self.Value0.ndim == 2:
            DeriType = 'matrix'
        elif is_vector(self.Value0):
            DeriType = 'vector'
        elif is_scalar(self.Value0):
            DeriType = 'scalar'
        else:
            raise TypeError(f"Cant deduce derivative type of value {self.Value0}")

        self.DeriType = DeriType

        # broadcast derivative
        if DiffVarType == 'scalar':
            match DeriType:
                case 'scalar':
                    self.DeriExprBc = self.DeriExpr * Ones(EqnSize)
                case 'vector':
                    if self.Value0.size != EqnSize:
                        raise ValueError(f"Vector derivative size {self.Value0.size} != Equation size {EqnSize}")
                    self.DeriExprBc = self.DeriExpr
                case 'matrix':
                    # self.DeriExprBc = self.DeriExpr
                    raise TypeError("Matrix derivative of scalar variables not supported!")
                case _:
                    raise TypeError(f"Derivative with value {self.Value0} of scalar variables not supported!")
        elif DiffVarType == 'vector':
            match DeriType:
                case 'scalar':
                    if self.DiffVarValue.size != EqnSize:
                        raise ValueError(f"Vector variable {self.DiffVar} size {self.DiffVarValue.size} != " +
                                         f"Equation size {EqnSize} in scalar derivative case.")
                    self.DeriExprBc = self.DeriExpr * Ones(EqnSize)
                case 'vector':
                    if self.Value0.size != EqnSize:
                        raise ValueError(f"Vector derivative size {self.Value0.size} != Equation size {EqnSize}")
                    if self.DiffVarValue.size != EqnSize:
                        raise ValueError(f"Vector variable {self.DiffVar} size {self.DiffVarValue.size} != " +
                                         f"Equation size {EqnSize} in vector derivative case.")
                    self.DeriExprBc = self.DeriExpr
                case 'matrix':
                    if self.Value0.shape[0] == EqnSize:
                        try:
                            self.Value0 @ DiffVarValue
                        except ValueError:
                            raise ValueError(f"Incompatible matrix derivative size {self.Value0.shape} " +
                                             f"and vector variable size {DiffVarValue.shape}.")
                    self.DeriExprBc = self.DeriExpr
                case _:
                    raise TypeError(f"Derivative with value {self.Value0} of vector variables not supported!")

        # parse sparse jac blocks, the address and expression
        self.ParseSp()
        # parse dense jac blocks, the address and expression
        self.ParseDen()

    @property
    def IsDeriNumber(self):
        return is_number(self.DeriExpr)

    def ParseSp(self):
        EqnSize = self.EqnAddr.stop - self.EqnAddr.start
        match self.DiffVarType:
            case 'vector':
                match self.DeriType:
                    case 'matrix':
                        warnings.warn("Sparse parser of matrix type jac block not implemented!")
                    case 'vector' | 'scalar':
                        self.SpEqnAddr = slice2array(self.EqnAddr)
                        if isinstance(self.DiffVar, iVar):
                            self.SpVarAddr = slice2array(self.VarAddr)
                        else:
                            if isinstance(self.DiffVar.index, slice):
                                VarArange = slice2array(self.VarAddr)[self.DiffVar.index]
                                self.SpVarAddr = VarArange
                            elif is_integer(self.DiffVar.index):
                                raise TypeError(f"Index of vector variable cant be integer!")
                            else:
                                raise TypeError(f"Index type {type(self.DiffVar.index)} not supported!")
                        self.SpDeriExpr = self.DeriExprBc
            case 'scalar':
                match self.DeriType:
                    case 'vector' | 'scalar':
                        self.SpEqnAddr = np.arange(self.EqnAddr.start, self.EqnAddr.stop)
                        if isinstance(self.DiffVar, iVar):
                            idx = self.VarAddr.start
                            self.SpVarAddr = np.array(EqnSize * [idx]).reshape((-1,)).astype(int)
                        else:
                            if isinstance(self.DiffVar.index, slice):
                                VarArange = slice2array(self.VarAddr)[self.DiffVar.index]
                                if VarArange.size > 1:
                                    raise ValueError(f"Length of scalar variable {self.DiffVar} > 1!")
                                else:
                                    idx = VarArange[0]
                                    self.SpVarAddr = np.array(EqnSize * [idx]).reshape((-1,)).astype(int)
                            elif is_integer(self.DiffVar.index):
                                idx = slice2array(self.VarAddr)[self.DiffVar.index]
                                self.SpVarAddr = np.array(EqnSize * [idx]).reshape((-1,)).astype(int)
                            else:
                                raise TypeError(f"Index type {type(self.DiffVar.index)} not supported!")
                        self.SpDeriExpr = self.DeriExprBc
        if self.SpEqnAddr.size != self.SpVarAddr.size:
            raise ValueError(f"Incompatible equation size {self.SpEqnAddr.size} of Equation {self.EqnName} " +
                             f"and variable size {self.SpVarAddr.size} of Variable {self.DiffVar}!")
        self.SpEleSize = self.SpEqnAddr.size

    def ParseDen(self):
        self.DenEqnAddr = self.EqnAddr
        match self.DiffVarType:
            case 'vector':
                match self.DeriType:
                    case 'matrix':
                        if isinstance(self.DiffVar, iVar):
                            self.DenVarAddr = self.VarAddr
                        else:
                            if isinstance(self.DiffVar.index, slice):
                                VarArange = slice2array(self.VarAddr)[self.DiffVar.index]
                                self.DenVarAddr = slice(VarArange[0], VarArange[-1] + 1)
                            elif is_integer(self.DiffVar.index):
                                raise TypeError(f"Index of vector variable cant be integer!")
                            else:
                                raise TypeError(f"Index type {type(self.DiffVar.index)} not supported!")
                        self.DenDeriExpr = self.DeriExprBc
                    case 'vector' | 'scalar':
                        self.DenEqnAddr = self.EqnAddr
                        if isinstance(self.DiffVar, iVar):
                            self.DenVarAddr = self.VarAddr
                        else:
                            if isinstance(self.DiffVar.index, slice):
                                VarArange = slice2array(self.VarAddr)[self.DiffVar.index]
                                self.DenVarAddr = slice(VarArange[0], VarArange[-1] + 1)
                            elif is_integer(self.DiffVar.index):
                                raise TypeError(f"Index of vector variable cant be integer!")
                            else:
                                raise TypeError(f"Index type {type(self.DiffVar.index)} not supported!")
                        self.DenDeriExpr = Diag(self.DeriExprBc)
            case 'scalar':
                match self.DeriType:
                    case 'vector' | 'scalar':
                        self.DenEqnAddr = self.EqnAddr
                        if isinstance(self.DiffVar, iVar):
                            self.DenVarAddr = self.VarAddr.start
                        else:
                            if isinstance(self.DiffVar.index, slice):
                                VarArange = slice2array(self.VarAddr)[self.DiffVar.index]
                                if VarArange.size > 1:
                                    raise ValueError(f"Length of scalar variable {self.DiffVar} > 1!")
                                else:
                                    self.DenVarAddr = VarArange[0]
                            elif is_integer(self.DiffVar.index):
                                idx = int(slice2array(self.VarAddr)[self.DiffVar.index])
                                self.DenVarAddr = idx
                            else:
                                raise TypeError(f"Index type {type(self.DiffVar.index)} not supported!")
                        self.DenDeriExpr = self.DeriExprBc

    def __repr__(self):
        return f"Jacblock with DeriExpr {self.DeriExpr.__repr__()}"


def slice2array(s: slice) -> np.ndarray:
    return np.arange(s.start, s.stop)




