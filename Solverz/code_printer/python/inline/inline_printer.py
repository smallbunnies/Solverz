from __future__ import annotations

from sympy.utilities.lambdify import _import, _module_present, _get_namespace

from Solverz.variable.variables import Vars
from Solverz.code_printer.python.utilities import *


# %%


def print_F(eqs: SymEquations):
    fp = print_func_prototype(eqs, 'F_')
    body = []
    body.extend(print_var(eqs))
    body.extend(print_param(eqs))
    body.extend(print_trigger(eqs))
    body.extend(print_eqn_assignment(eqs))
    temp = iVar('_F_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_J(eqs: SymEquations, sparse=False):
    fp = print_func_prototype(eqs, 'J_')
    # initialize temp
    temp = iVar('J_', internal_use=True)
    body = list()
    body.extend(print_var(eqs))
    body.extend(print_param(eqs))
    body.extend(print_trigger(eqs))
    if not sparse:
        body.append(Assignment(temp, zeros(eqs.eqn_size, eqs.vsize)))
        body.extend(print_J_blocks(eqs.jac, False))
        body.append(Return(temp))
    else:
        body.extend([Assignment(iVar('row', internal_use=True), SolList()),
                     Assignment(iVar('col', internal_use=True), SolList()),
                     Assignment(iVar('data', internal_use=True), SolList())])
        body.extend(print_J_blocks(eqs.jac, True))
        body.append(Return(coo_2_csc(eqs)))
    Jd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(Jd, fully_qualified_modules=False)


def print_J_blocks(jac: Jac, sparse: bool):
    eqn_declaration = []
    for eqn_name, jbs_row in jac.blocks.items():
        for var, jb in jbs_row.items():
            eqn_declaration.extend(print_J_block(jb,
                                                 sparse))
    return eqn_declaration


def print_J_block(jb: JacBlock, sparse: bool) -> List:
    if sparse:
        match jb.DeriType:
            case 'matrix':
                # return [Assignment(iVar('value_coo'), coo_array(rhs)),
                #         extend(iVar('row', internal_use=True), eqn_address),
                #         extend(iVar('col', internal_use=True), var_address),
                #         extend(iVar('data', internal_use=True), data)]
                raise NotImplementedError("Matrix parameters in sparse Jac not implemented yet!")
            case 'vector' | 'scalar':
                return [extend(iVar('row', internal_use=True), SolList(*jb.SpEqnAddr.tolist())),
                        extend(iVar('col', internal_use=True), SolList(*jb.SpVarAddr.tolist())),
                        extend(iVar('data', internal_use=True), jb.SpDeriExpr)]
    else:
        return [AddAugmentedAssignment(iVar('J_', internal_use=True)[jb.DenEqnAddr, jb.DenVarAddr],
                                       jb.DenDeriExpr)]


def made_numerical(eqn: SymEquations, y: Vars, sparse=False, output_code=False):
    """
    factory method of numerical equations
    """
    print(f"Printing numerical codes of {eqn.name}")
    eqn.FormJac(y)
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
