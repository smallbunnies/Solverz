from __future__ import annotations
from typing import Callable, Union, List
import builtins
from datetime import datetime
from numbers import Number

import numpy as np
from sympy.codegen.ast import Assignment, AddAugmentedAssignment
from sympy import pycode, symbols, Function, Symbol, Number as SymNumber, Expr
from sympy.codegen.ast import real, FunctionPrototype, FunctionDefinition, Return, FunctionCall
from sympy.utilities.lambdify import _import, _module_present, _get_namespace

from Solverz.equation.equations import Equations as SymEquations, FDAE as SymFDAE, DAE as SymDAE
from Solverz.equation.param import TimeSeriesParam
from Solverz.symboli_algebra.symbols import Var, SolDict, Para, idx
from Solverz.symboli_algebra.functions import zeros, CSC_array, Arange


# %%
def Solverzlambdify(funcstr, funcname, modules=None):
    """Convert a Solverz numerical f/g/F/J evaluation expression into a function that allows for fast
        numeric evaluation.
        """

    # If the user hasn't specified any modules, use what is available.
    if modules is None:
        try:
            _import("scipy")
        except ImportError:
            try:
                _import("numpy")
            except ImportError:
                # Use either numpy (if available) or python.math where possible.
                # XXX: This leads to different behaviour on different systems and
                #      might be the reason for irreproducible errors.
                modules = ["math", "mpmath", "sympy"]
            else:
                modules = ["numpy"]
        else:
            modules = ["numpy", "scipy"]

    # Get the needed namespaces.
    namespaces = []

    # Check for dict before iterating
    if isinstance(modules, (dict, str)) or not hasattr(modules, '__iter__'):
        namespaces.append(modules)
    else:
        # consistency check
        if _module_present('numexpr', modules) and len(modules) > 1:
            raise TypeError("numexpr must be the only item in 'modules'")
        namespaces += list(modules)
    # fill namespace with first having highest priority
    namespace = {}
    for m in namespaces[::-1]:
        buf = _get_namespace(m)
        namespace.update(buf)

    # Provide lambda expression with builtins, and compatible implementation of range
    namespace.update({'builtins': builtins, 'range': range})

    funclocals = {}
    current_time = datetime.now()
    filename = '<generated at-%s>' % current_time
    c = compile(funcstr, filename, 'exec')
    exec(c, namespace, funclocals)

    func = funclocals[funcname]
    # Apply the docstring
    src_str = funcstr
    # TODO: should collect and show the module imports from the code printers instead of the namespace
    func.__doc__ = (
        "Created with Solverz at \n\n"
        "{sig}\n\n"
        "Source code:\n\n"
        "{src}\n\n"
    ).format(sig=current_time, src=src_str)
    return func


def _print_var_parser(var_name, suffix: str, var_address):
    y = Var('y_' + suffix, internal_use=True)
    if suffix == '':
        return Assignment(Var(var_name + suffix), y[var_address])
    else:
        return Assignment(Var(var_name + '_tag_' + suffix), y[var_address])
    pass


def print_var(ae: SymEquations):
    var_declaration = []
    if hasattr(ae, 'nstep'):
        nstep = ae.nstep
    else:
        nstep = 0
    suffix = [''] + [f'{i}' for i in range(nstep)]
    for i in suffix:
        for var_name in ae.var_address.v.keys():
            var_address = ae.var_address[var_name]
            var_declaration.append(_print_var_parser(var_name, i, var_address))
    return var_declaration


def print_param(ae: SymEquations):
    param_declaration = []
    p = SolDict('p_')
    for symbol_name, symbol in ae.SYMBOLS.items():
        if isinstance(symbol, (Para, idx)):
            if isinstance(ae.PARAM[symbol_name], TimeSeriesParam):
                param_declaration.append(
                    Assignment(Para(symbol_name), FunctionCall(f'p_["{symbol_name}"].get_v_t', 't')))
            else:
                param_declaration.append(Assignment(Para(symbol_name), p[symbol_name]))
    return param_declaration


def _print_F_assignment(eqn_address, rhs):
    F_ = Var('F_', internal_use=True)
    return Assignment(F_[eqn_address], rhs)


def print_eqn_assignment(ae: SymEquations):
    temp = Var('F_', internal_use=True)
    eqn_declaration = [Assignment(temp, zeros(ae.eqn_size, ))]
    for eqn_name in ae.EQNs.keys():
        eqn_address = ae.a[eqn_name]
        eqn_declaration.append(_print_F_assignment(eqn_address, ae.EQNs[eqn_name].RHS))
    return eqn_declaration


def _parse_jac_eqn_address(eqn_address: slice, derivative_dim, sparse):
    if eqn_address.stop - eqn_address.start == 1:
        if derivative_dim == 1 or derivative_dim == 0:
            eqn_address = eqn_address.start if not sparse else SolList(eqn_address.start)
        else:
            if sparse:
                return Var(f'arange({eqn_address.start}, {eqn_address.stop})')[idx('value_coo.row')]
    else:
        if derivative_dim == 1 or derivative_dim == 0:
            eqn_address = Arange(eqn_address.start, eqn_address.stop)
        else:
            if sparse:
                return Var(f'arange({eqn_address.start}, {eqn_address.stop})')[idx('value_coo.row')]
    return eqn_address


def _parse_jac_var_address(var_address_slice: slice, derivative_dim, var_idx, sparse):
    if var_idx is not None:
        try:  # try to simplify the variable address in cases such as [1:10][0,1,2,3]
            temp = np.arange(var_address_slice.start, var_address_slice.stop)[var_idx]
            if isinstance(temp, (Number, SymNumber)):  # number a
                if not sparse:
                    var_address = temp
                else:
                    if derivative_dim == 0 or derivative_dim == 1:
                        var_address = SolList(temp)
                    else:
                        var_address = Var('array([' + str(temp) + '])')[idx('value_coo.col')]
            else:  # np.ndarray
                if np.all(np.diff(temp) == 1):  # list such as [1,2,3,4] can be viewed as slice [1:5]
                    if derivative_dim == 2:
                        if sparse:
                            return Var(f'arange({temp[0]}, {temp[-1] + 1})')[idx('value_coo.col')]
                        else:
                            var_address = slice(temp[0], temp[-1] + 1)
                    elif derivative_dim == 1 or derivative_dim == 0:
                        var_address = Arange(temp[0], temp[-1] + 1)
                else:  # arbitrary list such as [1,3,4,5,6,9]
                    if not sparse or derivative_dim<2:
                        var_address = SolList(*temp)
                    else:
                        var_address = Var('array(['+','.join([str(ele) for ele in temp])+'])')[idx('value_coo.col')]
        except (TypeError, IndexError):
            if isinstance(var_idx, str):
                var_idx = idx(var_idx)
            if not sparse or derivative_dim < 2:
                var_address = Var(f"arange({var_address_slice.start}, {var_address_slice.stop})")[var_idx]
            else:
                var_address = Var(f"arange({var_address_slice.start}, {var_address_slice.stop})")[var_idx[idx('value_coo.col')]]
    else:
        if derivative_dim == 2:
            if sparse:
                return Var(f'arange({var_address_slice.start}, {var_address_slice.stop})')[idx('value_coo.col')]
            else:
                var_address = var_address_slice
        elif derivative_dim == 1 or derivative_dim == 0:
            var_address = Arange(var_address_slice.start, var_address_slice.stop)
    return var_address


def _parse_jac_data(data_length, derivative_dim: int, rhs: Union[Expr, Number, SymNumber]):
    if derivative_dim == 2:
        return Var('value_coo.data')
    elif derivative_dim == 1:
        return rhs
    elif derivative_dim == 0:
        if isinstance(rhs, (Number, SymNumber)):  # if rhs is a number, then return length*[rhs]
            return data_length * SolList(rhs)
        else:  # if rhs produces np.ndarray then return length*(rhs).tolist()
            return data_length * tolist(rhs)


def print_J_block(eqn_address_slice, var_address_slice, derivative_dim, var_idx, rhs, sparse) -> List:
    if sparse:
        eqn_address = _parse_jac_eqn_address(eqn_address_slice,
                                             derivative_dim,
                                             True)
        var_address = _parse_jac_var_address(var_address_slice,
                                             derivative_dim,
                                             var_idx,
                                             True)
        # assign elements to sparse matrix can not be easily broadcast, so we have to parse the data
        data = _parse_jac_data(eqn_address_slice.stop - eqn_address_slice.start,
                               derivative_dim,
                               rhs)
        if derivative_dim < 2:
            return [extend(Var('row', internal_use=True), eqn_address),
                    extend(Var('col', internal_use=True), var_address),
                    extend(Var('data', internal_use=True), data)]
        else:
            return [Assignment(Var('value_coo'), coo_array(rhs)),
                    extend(Var('row', internal_use=True), eqn_address),
                    extend(Var('col', internal_use=True), var_address),
                    extend(Var('data', internal_use=True), data)]
    else:
        eqn_address = _parse_jac_eqn_address(eqn_address_slice,
                                             derivative_dim,
                                             False)
        var_address = _parse_jac_var_address(var_address_slice,
                                             derivative_dim,
                                             var_idx,
                                             False)
        return [AddAugmentedAssignment(Var('J_', internal_use=True)[eqn_address, var_address], rhs)]


def print_J_dense(ae: SymEquations):
    eqn_declaration = []
    for eqn_name, eqn in ae.EQNs.items():
        eqn_address_slice = ae.a[eqn_name]
        for var_name, eqndiff in eqn.derivatives.items():
            derivative_dim = eqndiff.dim
            if derivative_dim < 0:
                raise ValueError("Derivative dimension not assigned")
            var_address_slice = ae.var_address[eqndiff.diff_var_name]
            var_idx = eqndiff.var_idx
            rhs = eqndiff.RHS
            eqn_declaration.extend(print_J_block(eqn_address_slice,
                                                 var_address_slice,
                                                 derivative_dim,
                                                 var_idx,
                                                 rhs,
                                                 False))
    return eqn_declaration


def print_J_sparse(ae: SymEquations):
    eqn_declaration = []
    for eqn_name, eqn in ae.EQNs.items():
        eqn_address_slice = ae.a[eqn_name]
        for var_name, eqndiff in eqn.derivatives.items():
            derivative_dim = eqndiff.dim
            if derivative_dim < 0:
                raise ValueError("Derivative dimension not assigned")
            # if derivative_dim == 2:
            #     raise NotImplementedError("Not implemented")
            var_address_slice = ae.var_address[eqndiff.diff_var_name]
            var_idx = eqndiff.var_idx
            rhs = eqndiff.RHS
            eqn_declaration.extend(print_J_block(eqn_address_slice,
                                                 var_address_slice,
                                                 derivative_dim,
                                                 var_idx,
                                                 rhs,
                                                 True))
    return eqn_declaration


def _print_trigger_func(para_name, trigger_var: List[str]):
    return Assignment(Para(para_name),
                      FunctionCall(para_name + '_trigger_func', tuple(trigger_var)))


def print_trigger(ae: SymEquations):
    trigger_declaration = []
    for name, param in ae.PARAM.items():
        if param.triggerable:
            trigger_declaration.append(_print_trigger_func(name,
                                                           param.trigger_var))
    return trigger_declaration


def print_func_prototype(ae: SymEquations, func_name: str):
    t, y_, p_ = symbols('t y_ p_', real=True)
    if isinstance(ae, (SymDAE, SymFDAE)):
        args = [t, y_, p_]
    else:
        args = [y_, p_]
    xtra_args = []
    if hasattr(ae, 'nstep'):
        if ae.nstep > 0:
            xtra_args.extend([symbols('y_' + f'{i}', real=True) for i in range(ae.nstep)])

    fp = FunctionPrototype(real, func_name, args + xtra_args)
    return fp


def print_F(ae: SymEquations):
    fp = print_func_prototype(ae, 'F_')
    body = []
    body.extend(print_var(ae))
    body.extend(print_param(ae))
    body.extend(print_trigger(ae))
    body.extend(print_eqn_assignment(ae))
    temp = Var('F_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_J(ae: SymEquations, sparse=False):
    fp = print_func_prototype(ae, 'J_')
    # initialize temp
    temp = Var('J_', internal_use=True)
    body = list()
    body.extend(print_var(ae))
    body.extend(print_param(ae))
    body.extend(print_trigger(ae))
    if not sparse:
        body.append(Assignment(temp, zeros(ae.eqn_size, ae.vsize)))
        body.extend(print_J_dense(ae))
        body.append(Return(temp))
    else:
        body.extend([Assignment(Var('row', internal_use=True), SolList()),
                     Assignment(Var('col', internal_use=True), SolList()),
                     Assignment(Var('data', internal_use=True), SolList())])
        body.extend(print_J_sparse(ae))
        body.append(Return(coo_2_csc(ae)))
    Jd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(Jd, fully_qualified_modules=False)


class coo_2_csc(Symbol):

    def __new__(cls, ae: SymEquations):
        obj = Symbol.__new__(cls, f'coo_2_csc: {ae.name}')
        obj.eqn_size = ae.eqn_size
        obj.vsize = ae.vsize
        return obj

    def _numpycode(self, printer, **kwargs):
        return f'coo_array((data, (row,col)), ({self.eqn_size}, {self.vsize})).tocsc()'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class coo_array(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) > 1:
            raise ValueError(f"Solverz' coo_array object accepts only one inputs.")

    def _numpycode(self, printer, **kwargs):
        return f'coo_array({printer._print(self.args[0])})'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class extend(Function):

    def _numpycode(self, printer, **kwargs):
        return f'{printer._print(self.args[0])}.extend({printer._print(self.args[1])})'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class SolList(Function):

    # @classmethod
    # def eval(cls, *args):
    #     if any([not isinstance(arg, Number) for arg in args]):
    #         raise ValueError(f"Solverz' list object accepts only number inputs.")

    def _numpycode(self, printer, **kwargs):
        return r'[' + ', '.join([printer._print(arg) for arg in self.args]) + r']'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

class Array(Function):


    def _numpycode(self, printer, **kwargs):
        return r'array([' + ', '.join([printer._print(arg) for arg in self.args]) + r'])'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

class tolist(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) != 1:
            raise ValueError(f"Solverz' tolist function accepts only one input.")

    def _numpycode(self, printer, **kwargs):
        return r'((' + printer._print(self.args[0]) + r').tolist())'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class inner_F(Symbol):

    def __new__(cls, ae: SymEquations, *args):
        obj = Symbol.__new__(cls, f'innerF: {ae.name}')
        obj.inner_F_args = args
        return obj

    def _numpycode(self, printer, **kwargs):
        return r'inner_F(' + ',\n '.join([printer._print(arg) for arg in self.inner_F_args]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)
