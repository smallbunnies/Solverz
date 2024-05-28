from __future__ import annotations

from sympy.utilities.lambdify import _import, _module_present, _get_namespace

from Solverz.code_printer.python.utilities import *


# %%


def print_F(ae: SymEquations):
    fp = print_func_prototype(ae, 'F_')
    body = []
    body.extend(print_var(ae))
    body.extend(print_param(ae))
    body.extend(print_trigger(ae))
    body.extend(print_eqn_assignment(ae))
    temp = iVar('_F_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_J(ae: SymEquations, sparse=False):
    fp = print_func_prototype(ae, 'J_')
    # initialize temp
    temp = iVar('J_', internal_use=True)
    body = list()
    body.extend(print_var(ae))
    body.extend(print_param(ae))
    body.extend(print_trigger(ae))
    if not sparse:
        body.append(Assignment(temp, zeros(ae.eqn_size, ae.vsize)))
        body.extend(print_J_dense(ae))
        body.append(Return(temp))
    else:
        body.extend([Assignment(iVar('row', internal_use=True), SolList()),
                     Assignment(iVar('col', internal_use=True), SolList()),
                     Assignment(iVar('data', internal_use=True), SolList())])
        body.extend(print_J_sparse(ae))
        body.append(Return(coo_2_csc(ae)))
    Jd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(Jd, fully_qualified_modules=False)


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


def print_J_block(eqn_address_slice, var_address_slice, derivative_dim, var_idx, rhs, sparse,
                  rhs_v_dtpe='array') -> List:
    if sparse:
        eqn_address = _parse_jac_eqn_address(eqn_address_slice,
                                             derivative_dim,
                                             True)
        var_address = _parse_jac_var_address(var_address_slice,
                                             derivative_dim,
                                             var_idx,
                                             True,
                                             eqn_address_slice.stop - eqn_address_slice.start)
        # assign elements to sparse matrix can not be easily broadcast, so we have to parse the data
        data = _parse_jac_data(eqn_address_slice.stop - eqn_address_slice.start,
                               derivative_dim,
                               rhs,
                               rhs_v_dtpe)
        if derivative_dim < 2:
            return [extend(iVar('row', internal_use=True), eqn_address),
                    extend(iVar('col', internal_use=True), var_address),
                    extend(iVar('data', internal_use=True), data)]
        else:
            return [Assignment(iVar('value_coo'), coo_array(rhs)),
                    extend(iVar('row', internal_use=True), eqn_address),
                    extend(iVar('col', internal_use=True), var_address),
                    extend(iVar('data', internal_use=True), data)]
    else:
        eqn_address = _parse_jac_eqn_address(eqn_address_slice,
                                             derivative_dim,
                                             False)
        var_address = _parse_jac_var_address(var_address_slice,
                                             derivative_dim,
                                             var_idx,
                                             False)
        return [AddAugmentedAssignment(iVar('J_', internal_use=True)[eqn_address, var_address], rhs)]


def print_J_sparse(ae: SymEquations):
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
            rhs_v_type = eqndiff.v_type
            eqn_declaration.extend(print_J_block(eqn_address_slice,
                                                 var_address_slice,
                                                 derivative_dim,
                                                 var_idx,
                                                 rhs,
                                                 True,
                                                 rhs_v_type))
    return eqn_declaration


def _parse_jac_eqn_address(eqn_address: slice, derivative_dim, sparse):
    if eqn_address.stop - eqn_address.start == 1:
        if derivative_dim == 1 or derivative_dim == 0:
            eqn_address = eqn_address.start if not sparse else SolList(eqn_address.start)
        else:
            if sparse:
                return iVar(f'arange({eqn_address.start}, {eqn_address.stop})')[idx('value_coo.row')]
    else:
        if derivative_dim == 1 or derivative_dim == 0:
            eqn_address = Arange(eqn_address.start, eqn_address.stop)
        else:
            if sparse:
                return iVar(f'arange({eqn_address.start}, {eqn_address.stop})')[idx('value_coo.row')]
    return eqn_address


def _parse_jac_var_address(var_address_slice: slice,
                           derivative_dim,
                           var_idx,
                           sparse,
                           eqn_length: int = 1):
    if var_idx is not None:
        try:  # try to simplify the variable address in cases such as [1:10][0,1,2,3]
            temp = np.arange(var_address_slice.start, var_address_slice.stop)[var_idx]
            if isinstance(temp, (Number, SymNumber)):  # number a
                if not sparse:
                    var_address = temp
                else:
                    if derivative_dim == 0:  # For the derivative of omega[0] in eqn omega-omega[0] where omega is a vector
                        var_address = eqn_length * SolList(temp)
                    elif derivative_dim == 1:
                        var_address = SolList(temp)
                    else:
                        var_address = iVar('array([' + str(temp) + '])')[idx('value_coo.col')]
            else:  # np.ndarray
                if np.all(np.diff(temp) == 1):  # list such as [1,2,3,4] can be viewed as slice [1:5]
                    if derivative_dim == 2:
                        if sparse:
                            return iVar(f'arange({temp[0]}, {temp[-1] + 1})')[idx('value_coo.col')]
                        else:
                            var_address = slice(temp[0], temp[-1] + 1)
                    elif derivative_dim == 1 or derivative_dim == 0:
                        var_address = Arange(temp[0], temp[-1] + 1)
                else:  # arbitrary list such as [1,3,4,5,6,9]
                    if not sparse or derivative_dim < 2:
                        var_address = SolList(*temp)
                    else:
                        var_address = iVar('array([' + ','.join([str(ele) for ele in temp]) + '])')[
                            idx('value_coo.col')]
        except (TypeError, IndexError):
            if isinstance(var_idx, str):
                var_idx = idx(var_idx)
            if not sparse or derivative_dim < 2:
                var_address = iVar(f"arange({var_address_slice.start}, {var_address_slice.stop})")[var_idx]
            else:
                var_address = iVar(f"arange({var_address_slice.start}, {var_address_slice.stop})")[
                    var_idx[idx('value_coo.col')]]
    else:
        if derivative_dim == 2:
            if sparse:
                return iVar(f'arange({var_address_slice.start}, {var_address_slice.stop})')[idx('value_coo.col')]
            else:
                var_address = var_address_slice
        elif derivative_dim == 1 or derivative_dim == 0:
            var_address = Arange(var_address_slice.start, var_address_slice.stop)
    return var_address


def _parse_jac_data(data_length, derivative_dim: int, rhs: Union[Expr, Number, SymNumber], rhs_v_type='array'):
    if derivative_dim == 2:
        return iVar('value_coo.data')
    elif derivative_dim == 1:
        return rhs
    elif derivative_dim == 0:
        if rhs_v_type == 'Number':  # if rhs is a number, then return length*[rhs]
            return data_length * SolList(rhs)
        elif rhs_v_type == 'array':  # if rhs produces np.ndarray then return length*rhs.tolist()
            return data_length * tolist(rhs)


def made_numerical(eqn: SymEquations, *xys, sparse=False, output_code=False):
    """
    factory method of numerical equations
    """
    print(f"Printing numerical codes of {eqn.name}")
    eqn.assign_eqn_var_address(*xys)
    code_F = print_F(eqn)
    code_J = print_J(eqn, sparse)
    custom_func = dict()
    custom_func.update(numerical_interface)
    custom_func.update(parse_trigger_fun(eqn))
    F = Solverzlambdify(code_F, 'F_', modules=[custom_func, 'numpy'])
    J = Solverzlambdify(code_J, 'J_', modules=[custom_func, 'numpy'])
    p = parse_p(eqn)
    print('Complete!')
    if isinstance(eqn, SymAE) and not isinstance(eqn, SymFDAE):
        num_eqn = nAE(F, J, p)
    elif isinstance(eqn, SymFDAE):
        num_eqn = nFDAE(F, J, p, eqn.nstep)
    elif isinstance(eqn, SymDAE):
        num_eqn = nDAE(eqn.M, F, J, p)
    else:
        raise ValueError(f'Unknown equation type {type(eqn)}')
    if output_code:
        return num_eqn, {'F': code_F, 'J': code_J}
    else:
        return num_eqn


def parse_name_space(modules=None):
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
    return namespace


def Solverzlambdify(funcstr, funcname, modules=None):
    """Convert a Solverz numerical f/g/F/J evaluation expression into a function that allows for fast
        numeric evaluation.
        """
    namespace = parse_name_space(modules)

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


class SolList(Function):

    # @classmethod
    # def eval(cls, *args):
    #     if any([not isinstance(arg, Number) for arg in args]):
    #         raise ValueError(f"Solverz' list object accepts only number inputs.")

    def _numpycode(self, printer, **kwargs):
        return r'[' + ', '.join([printer._print(arg) for arg in self.args]) + r']'

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
