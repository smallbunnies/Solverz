from __future__ import annotations

from Solverz.code_printer.python.utilities import *


# %%


def print_J_numba(ae: SymEquations):
    fp = print_func_prototype(ae, 'J_')
    body = []
    var_assignments, var_list = print_var(ae, numba_printer=True)
    body.extend(var_assignments)
    param_assignments, param_list = print_param(ae, numba_printer=True)
    body.extend(param_assignments)
    body.extend(print_trigger(ae))
    body.extend([Assignment(iVar('data', internal_use=True),
                            FunctionCall('inner_J', [symbols('_data_', real=True)] + var_list + param_list))])
    body.extend([Return(coo_2_csc(ae))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_J(ae: SymEquations, *xys):
    var_assignments, var_list = print_var(ae, numba_printer=True)
    param_assignments, param_list = print_param(ae, numba_printer=True)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))
    fp = FunctionPrototype(real, 'inner_J', [symbols('_data_', real=True)] + args)
    body = []
    row, col, jac_address = parse_jac_address(ae, *xys)
    data = np.zeros_like(row, dtype=np.float64)
    # temp = iVar('data_', internal_use=True)
    # body.extend([Assignment(temp, zeros(jac_address.total_size, ))])
    code_sub_inner_J_blocks = []
    count = 0
    for jac_ in jac_address.object_list:
        eqn_name, var_name = jac_.split("@@@")
        derivative = ae.EQNs[eqn_name].derivatives[var_name]
        rhs = derivative.RHS
        if isinstance(rhs, (Number, SymNumber)):
            data[jac_address[jac_]] = rhs
        else:
            body.append(Assignment(iVar('_data_', internal_use=True)[jac_address[jac_]],
                                   FunctionCall(f'inner_J{int(count)}', list(derivative.SYMBOLS.values()))))
            args1 = []
            for var in derivative.SYMBOLS.keys():
                args1.append(symbols(var, real=True))

            fp1 = FunctionPrototype(real, f'inner_J{int(count)}', args1)
            body1 = [Return(rhs)]
            fd1 = FunctionDefinition.from_FunctionPrototype(fp1, body1)
            code_sub_inner_J_blocks.append(pycode(fd1, fully_qualified_modules=False))
            count += 1
    temp = iVar('_data_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return {'code_inner_J': pycode(fd, fully_qualified_modules=False),
            'code_sub_inner_J': code_sub_inner_J_blocks,
            'row': row,
            'col': col,
            'data': data}


def print_F_numba(ae: SymEquations):
    fp = print_func_prototype(ae, 'F_')
    body = []
    var_assignments, var_list = print_var(ae, numba_printer=True)
    body.extend(var_assignments)
    param_assignments, param_list = print_param(ae, numba_printer=True)
    body.extend(param_assignments)
    body.extend(print_trigger(ae))
    body.extend(
        [Return(FunctionCall('inner_F', [symbols('_F_', real=True)] + var_list + param_list))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_F(ae: SymEquations):
    var_assignments, var_list = print_var(ae, numba_printer=True)
    param_assignments, param_list = print_param(ae, numba_printer=True)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))
    fp = FunctionPrototype(real, 'inner_F', [symbols('_F_', real=True)] + args)
    body = []
    body.extend(print_eqn_assignment(ae, True))
    temp = iVar('_F_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_sub_inner_F(ae: SymEquations):
    code_blocks = []
    count = 0
    for eqn_name in ae.EQNs.keys():
        eqn = ae.EQNs[eqn_name]
        args = []
        for var in eqn.SYMBOLS.keys():
            args.append(symbols(var, real=True))
        fp = FunctionPrototype(real, f'inner_F{count}', args)
        body = [Return(eqn.RHS)]
        fd = FunctionDefinition.from_FunctionPrototype(fp, body)
        count = count + 1
        code_blocks.append(pycode(fd, fully_qualified_modules=False))
    return code_blocks


def parse_jac_address(eqns: SymEquations, *xys):
    if isinstance(eqns, SymAE):
        gy = eqns.gy(*xys)
    elif isinstance(eqns, SymDAE):
        gy = eqns.fy(0, *xys)
        if len(eqns.g_list) > 0:
            gy.extend(eqns.gy(0, *xys))
    else:
        raise NotImplementedError(f"Unknown equation type {type(eqns)}")

    row = np.array([], dtype=int)
    col = np.array([], dtype=int)
    jac_block_address = Address()
    for gy_tuple in gy:
        eqn_name = gy_tuple[0]
        # eqn_diffs: Dict[str, EqnDiff] = eqns.EQNs[eqn_name].derivatives
        var_name = gy_tuple[1]
        eqndiff = gy_tuple[2]
        diff_var = eqndiff.diff_var
        diff_var_name = eqndiff.diff_var.name
        value = gy_tuple[3]
        eqn_address = eqns.a[eqn_name]
        var_address = eqns.var_address[var_name]
        if isinstance(value, (np.ndarray, sparray)):
            if value.ndim == 2:  # matrix
                raise TypeError("Two-dimensional array not applicable for numba printer!\n  Try rewrite the Equations!")
            elif value.ndim == 1 and value.shape[0] != 1:  # vector
                num_jac_element = value.shape[0]
                if num_jac_element != eqn_address.stop - eqn_address.start:
                    raise ValueError("Number of jac block elements not compatible with equation length!")
            elif value.ndim == 1 and value.shape[0] == 1:  # scalar in np.ndarray for example array([0.0])
                num_jac_element = eqn_address.stop - eqn_address.start
            else:
                raise ValueError("Unknown derivative value dimension type!")
        elif isinstance(value, (Number, SymNumber)):
            num_jac_element = eqn_address.stop - eqn_address.start
        else:
            raise ValueError(f"Unknown derivative data type {type(value)}!")
        eqn_address_range = np.arange(eqn_address.start, eqn_address.stop)
        row = np.append(row, eqn_address_range)
        if isinstance(diff_var, IdxSymBasic):
            index = diff_var.index
            if not isinstance(index, (int, slice, np.integer)):
                raise TypeError(
                    f"Index type {type(diff_var.index)} not applicable for numba printer!\n Try rewrite the Variable!")
            else:
                # reshape is to convert float/integer to 1-dim numpy.ndarray
                var_address_range = np.array(np.arange(var_address.start, var_address.stop)[index]).reshape((-1,))
                if len(var_address_range) < len(eqn_address_range):
                    warnings.warn(
                        f'Address of variable {diff_var_name} (length={len(var_address_range)}) shorter than equation address of {eqn_name} (length={len(eqn_address_range)}). Please check the variable address and equation address of this part.')
                    var_address_range = np.array(len(eqn_address_range) * (var_address_range.tolist()))
        elif isinstance(diff_var, iVar):
            var_address_range = np.arange(var_address.start, var_address.stop)
        if len(var_address_range) != len(eqn_address_range):
            raise ValueError('Equation address range is different from variable address range')
        col = np.append(col, var_address_range)
        jac_block_address.add(f'{eqn_name}@@@{diff_var_name}', num_jac_element)

    assert len(row) == len(col) == jac_block_address.total_size

    return row, col, jac_block_address


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





