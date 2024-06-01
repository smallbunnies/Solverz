from __future__ import annotations

from Solverz.code_printer.python.utilities import *


# %%


def print_J(eqs: SymEquations):
    fp = print_F_J_prototype(eqs.__class__.__name__, 
                             'J_')
    body = []
    var_assignments, var_list = print_var(eqs.var_address, 
                                          nstep=eqs.nstep)
    body.extend(var_assignments)
    param_assignments, param_list = print_param(eqs.PARAM)
    body.extend(param_assignments)
    body.extend(print_trigger(eqs.PARAM))
    body.extend([Assignment(iVar('data', internal_use=True),
                            FunctionCall('inner_J', [symbols('_data_', real=True)] + var_list + param_list))])
    body.extend([Return(coo_2_csc(eqs))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_J(eqs: SymEquations):
    var_assignments, var_list = print_var(eqs.var_address,
                                          eqs.nstep)
    param_assignments, param_list = print_param(eqs.PARAM)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))
    fp = FunctionPrototype(real, 'inner_J', [symbols('_data_', real=True)] + args)
    body = []
    jac = eqs.jac

    code_sub_inner_J_blocks = []
    count = 0
    addr_by_ele_0 = 0
    for eqn_name, jbs_row in jac.blocks.items():
        for var, jb in jbs_row.items():
            rhs = jb.SpDeriExpr
            SymbolsInDeri_ = list(Eqn(f'temp' + eqn_name + var.name, rhs).SYMBOLS.values())
            # add real assumption
            SymbolsInDeri = [symbols(arg.name, real=True) for arg in SymbolsInDeri_]
            addr_by_ele = slice(addr_by_ele_0, addr_by_ele_0 + jb.SpEleSize)
            if not jb.IsDeriNumber:
                # _data_[0:1] = inner_J0(t1, x)
                body.append(Assignment(iVar('_data_', internal_use=True)[addr_by_ele],
                                       FunctionCall(f'inner_J{int(count)}', SymbolsInDeri)))

                # def inner_J0(t1, x):
                #     return -t1 * pi * cos(pi * x) + 1
                fp1 = FunctionPrototype(real, f'inner_J{int(count)}', SymbolsInDeri)
                body1 = [Return(rhs)]
                fd1 = FunctionDefinition.from_FunctionPrototype(fp1, body1)
                code_sub_inner_J_blocks.append(pycode(fd1, fully_qualified_modules=False))
                count += 1
            addr_by_ele_0 += jb.SpEleSize
    temp = iVar('_data_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return {'code_inner_J': pycode(fd, fully_qualified_modules=False),
            'code_sub_inner_J': code_sub_inner_J_blocks}


def print_F(eqs: SymEquations):
    fp = print_F_J_prototype(eqs.__class__.__name__,
                             'F_')
    body = []
    var_assignments, var_list = print_var(eqs.var_address,
                                          eqs.nstep)
    body.extend(var_assignments)
    param_assignments, param_list = print_param(eqs.PARAM)
    body.extend(param_assignments)
    body.extend(print_trigger(eqs.PARAM))
    body.extend(
        [Return(FunctionCall('inner_F', [symbols('_F_', real=True)] + var_list + param_list))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_F(eqs: SymEquations):
    var_assignments, var_list = print_var(eqs.var_address,
                                          eqs.nstep)
    param_assignments, param_list = print_param(eqs.PARAM)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))
    fp = FunctionPrototype(real, 'inner_F', [symbols('_F_', real=True)] + args)
    body = []
    body.extend(print_eqn_assignment(eqs.EQNs,
                                     eqs.a,
                                     True))
    temp = iVar('_F_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_sub_inner_F(eqs: SymEquations):
    code_blocks = []
    count = 0
    for eqn_name in eqs.EQNs.keys():
        eqn = eqs.EQNs[eqn_name]
        args = []
        for var in eqn.SYMBOLS.keys():
            args.append(symbols(var, real=True))
        fp = FunctionPrototype(real, f'inner_F{count}', args)
        body = [Return(eqn.RHS)]
        fd = FunctionDefinition.from_FunctionPrototype(fp, body)
        count = count + 1
        code_blocks.append(pycode(fd, fully_qualified_modules=False))
    return code_blocks


class coo_2_csc(Symbol):

    def __new__(cls, eqs: SymEquations):
        obj = Symbol.__new__(cls, f'coo_2_csc: {eqs.name}')
        obj.eqn_size = eqs.eqn_size
        obj.vsize = eqs.vsize
        return obj

    def _numpycode(self, printer, **kwargs):
        return f'coo_array((data, (row,col)), ({self.eqn_size}, {self.vsize})).tocsc()'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)
