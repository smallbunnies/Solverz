from __future__ import annotations

from Solverz.code_printer.python.utilities import *


# %%
def print_Hvp(eqs_type: str,
              eqn_size: int,
              var_addr: Address,
              PARAM: Dict[str, ParamBase],
              nstep: int = 0):
    if eqn_size != var_addr.total_size:
        raise ValueError(f"Hvp matrix, with size ({eqn_size}*{var_addr.total_size}), not square")
    fp = print_Hvp_prototype(eqs_type,
                             'Hvp_',
                             nstep=nstep)
    body = []
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    body.extend(var_assignments)
    param_assignments, param_list = print_param(PARAM)
    body.extend(param_assignments)
    body.extend(print_trigger(PARAM))
    args = [symbols('_data_hvp', real=True), symbols('v_', real=True)] + var_list + param_list
    body.extend([Assignment(iVar('data_hvp', internal_use=True), FunctionCall('inner_Hvp', args))])
    body.extend([Return(coo_2_csc_hvp(eqn_size, var_addr.total_size))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_Hvp(var_addr: Address,
                    PARAM: Dict[str, ParamBase],
                    hvp: Hvp,
                    nstep: int = 0):
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    param_assignments, param_list = print_param(PARAM)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))
    fp = FunctionPrototype(real, 'inner_Hvp',
                           [symbols('_data_hvp', real=True), symbols('v_', real=True)] + args)
    body = []

    code_sub_inner_Hvp_blocks = []
    count = 0
    addr_by_ele_0 = 0
    for eqn_name, jbs_row in hvp.blocks_sorted.items():
        for var, jb in jbs_row.items():
            rhs = jb.SpDeriExpr
            SymbolsInDeri_ = list(Eqn(f'temp' + eqn_name + var.name, rhs).SYMBOLS.values())
            # add real assumption
            SymbolsInDeri = [symbols(arg.name, real=True) for arg in SymbolsInDeri_]
            addr_by_ele = slice(addr_by_ele_0, addr_by_ele_0 + jb.SpEleSize)
            if not jb.IsDeriNumber:
                # _data_[0:1] = inner_Hvp0(v_, t1, x)
                body.append(Assignment(iVar('_data_hvp', internal_use=True)[addr_by_ele],
                                       FunctionCall(f'inner_Hvp{int(count)}', SymbolsInDeri)))

                # def inner_Hvp0(v_, t1, x):
                #     return -t1 * pi * cos(pi * x) + 1
                fp1 = FunctionPrototype(real, f'inner_Hvp{int(count)}', SymbolsInDeri)
                body1 = [Return(rhs)]
                fd1 = FunctionDefinition.from_FunctionPrototype(fp1, body1)
                code_sub_inner_Hvp_blocks.append(pycode(fd1, fully_qualified_modules=False))
                count += 1
            addr_by_ele_0 += jb.SpEleSize
    temp = iVar('_data_hvp', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return {'code_inner_Hvp': pycode(fd, fully_qualified_modules=False),
            'code_sub_inner_Hvp': code_sub_inner_Hvp_blocks}


def print_J(eqs_type: str,
            eqn_size: int,
            var_addr: Address,
            PARAM: Dict[str, ParamBase],
            nstep: int = 0):
    if eqn_size != var_addr.total_size:
        raise ValueError(f"Jac matrix, with size ({eqn_size}*{var_addr.total_size}), not square")
    fp = print_F_J_prototype(eqs_type,
                             'J_',
                             nstep)
    body = []
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    body.extend(var_assignments)
    param_assignments, param_list = print_param(PARAM)
    body.extend(param_assignments)
    body.extend(print_trigger(PARAM))
    body.extend([Assignment(iVar('data', internal_use=True),
                            FunctionCall('inner_J', [symbols('_data_', real=True)] + var_list + param_list))])
    body.extend([Return(coo_2_csc(eqn_size, var_addr.total_size))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_J(var_addr: Address,
                  PARAM: Dict[str, ParamBase],
                  jac: Jac,
                  nstep: int = 0):
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    param_assignments, param_list = print_param(PARAM)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))
    fp = FunctionPrototype(real, 'inner_J', [symbols('_data_', real=True)] + args)
    body = []

    code_sub_inner_J_blocks = []
    count = 0
    addr_by_ele_0 = 0
    for eqn_name, jbs_row in jac.blocks_sorted.items():
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


def print_F(eqs_type: str,
            var_addr: Address,
            PARAM: Dict[str, ParamBase],
            nstep: int = 0):
    fp = print_F_J_prototype(eqs_type,
                             'F_',
                             nstep)
    body = []
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    body.extend(var_assignments)
    param_assignments, param_list = print_param(PARAM)
    body.extend(param_assignments)
    body.extend(print_trigger(PARAM))
    body.extend(
        [Return(FunctionCall('inner_F', [symbols('_F_', real=True)] + var_list + param_list))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_F(EQNs: Dict[str, Eqn],
                  EqnAddr: Address,
                  var_addr: Address,
                  PARAM: Dict[str, ParamBase],
                  nstep: int = 0):
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    param_assignments, param_list = print_param(PARAM)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))
    fp = FunctionPrototype(real, 'inner_F', [symbols('_F_', real=True)] + args)
    body = []
    body.extend(print_eqn_assignment(EQNs,
                                     EqnAddr,
                                     True))
    temp = iVar('_F_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_sub_inner_F(EQNs: Dict[str, Eqn]):
    code_blocks = []
    count = 0
    for eqn_name in EQNs.keys():
        eqn = EQNs[eqn_name]
        args = []
        for var in eqn.SYMBOLS.keys():
            args.append(symbols(var, real=True))
        fp = FunctionPrototype(real, f'inner_F{count}', args)
        body = [Return(eqn.RHS)]
        fd = FunctionDefinition.from_FunctionPrototype(fp, body)
        count = count + 1
        code_blocks.append(pycode(fd, fully_qualified_modules=False))
    return code_blocks
