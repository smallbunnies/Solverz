from __future__ import annotations

import warnings
from numbers import Number
from typing import Union, List, Dict, Tuple
from copy import deepcopy

import numpy as np
from Solverz.equation.hvp import Hvp
from sympy import Symbol, Integer, Expr, Number as SymNumber
from scipy.sparse import csc_array, coo_array
# from cvxopt import spmatrix, matrix

from Solverz.equation.eqn import Eqn, Ode, EqnDiff
from Solverz.equation.param import ParamBase, Param, IdxParam, TimeSeriesParam
from Solverz.sym_algebra.symbols import iVar, idx, IdxVar, Para, iAliasVar
from Solverz.sym_algebra.functions import Slice, Mat_Mul, Diag, SpDiag
from Solverz.variable.variables import Vars
from Solverz.utilities.address import Address, combine_Address
from Solverz.utilities.type_checker import is_integer
from Solverz.num_api.Array import Array
from Solverz.equation.jac import Jac, JacBlock, is_constant_matrix_deri
from Solverz.utilities.miscellaneous import rearrange_list


class Equations:

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 matrix_container='scipy'):
        self.name = name

        self.EQNs: Dict[str, Eqn] = dict()
        self.SYMBOLS: Dict[str, Symbol] = dict()
        self.a = Address()  # equation address
        self.var_address = Address()  # variable address
        self.esize: Dict[str, int] = dict()  # size of each equation
        self.vsize: int = 0  # size of variables
        self.f_list = []
        self.g_list = []
        self.matrix_container = matrix_container
        self.PARAM: Dict[str, ParamBase] = dict()
        self.triggerable_quantity: Dict[str, str] = dict()
        self.jac_element_address = Address()
        self.jac: Jac = Jac()
        self.nstep = 0

        if isinstance(eqn, Eqn):
            eqn = [eqn]

        for eqn_ in eqn:
            self.add_eqn(eqn_)

    def add_eqn(self, eqn: Eqn):
        if eqn.name in self.EQNs.keys():
            raise ValueError(f"Equation {eqn.name} already defined!")
        self.EQNs.update({eqn.name: eqn})
        self.SYMBOLS.update(eqn.SYMBOLS)
        self.a.add(eqn.name)
        if isinstance(eqn, Eqn) and not isinstance(eqn, Ode):
            self.g_list = self.g_list + [eqn.name]
        elif isinstance(eqn, Ode):
            self.f_list = self.f_list + [eqn.name]

        for symbol_ in eqn.SYMBOLS.values():
            if isinstance(symbol_, Para):
                # this is not fully initialize of Parameters, please use param_initializer
                self.PARAM[symbol_.name] = Param(symbol_.name,
                                                 value=symbol_.value,
                                                 dim=symbol_.dim)
            elif isinstance(symbol_, iAliasVar):
                self.PARAM[symbol_.name] = Param(symbol_.name,
                                                 value=symbol_.value,
                                                 dim=symbol_.dim,
                                                 is_alias=True)
            elif isinstance(symbol_, idx):
                self.PARAM[symbol_.name] = IdxParam(symbol_.name, value=symbol_.value)

        self.EQNs[eqn.name].derive_derivative()

    def assign_eqn_var_address(self, *args):
        pass

    def Fy(self,
           y,
           eqn_list: List[str] = None,
           var_list: List[str] = None) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
        pass

    def _check_no_timevar_sparse_matrices(self):
        """Backstop check: reject any time-varying sparse ``dim=2``
        ``Param`` in the equation system.

        The primary defence lives in :class:`ParamBase.__init__` /
        :class:`TimeSeriesParam.__init__`, which reject the offending
        shape at the point of construction. This method re-checks at
        ``FormJac`` time to cover the unusual paths where a Param is
        built via ``__new__`` without going through ``__init__``, or
        where its ``triggerable`` flag is set after construction. A
        wrong model that slips past both defences would silently
        produce incorrect Jacobians because every downstream code
        path (the legacy ``MatVecMul`` CSC decomposition, the 0.8.1
        ``Mat_Mul`` ``SolCF.csc_matvec`` fast path, and the
        mutable-matrix Jacobian scatter-add kernel) caches the sparse
        matrix's ``.data`` / ``.indices`` / ``.indptr`` arrays at
        model-build time.
        """
        for name, p in self.PARAM.items():
            if not (getattr(p, 'dim', 0) == 2
                    and getattr(p, 'sparse', False)):
                continue
            is_ts = isinstance(p, TimeSeriesParam)
            is_trig = getattr(p, 'triggerable', False)
            if not (is_ts or is_trig):
                continue
            kind = 'time-series' if is_ts else 'triggerable'
            raise NotImplementedError(
                f"Parameter {name!r} is a {kind} sparse ``dim=2`` "
                f"parameter. Time-varying sparse matrices are not "
                f"supported by Solverz because the code generator "
                f"caches the matrix's CSC fields at model-build "
                f"time; a runtime {kind} update would silently be "
                f"ignored by every downstream consumer (legacy "
                f"``MatVecMul``, ``Mat_Mul`` fast path, and the "
                f"mutable-matrix Jacobian scatter-add). Rewrite the "
                f"equation in explicit element-wise form (one scalar "
                f"``Eqn`` per row) or use a dense ``dim=2`` "
                f"parameter (``sparse=False``) — the fallback scipy "
                f"path re-evaluates the full expression on every "
                f"call and tolerates updates."
            )

    def _warn_dense_matmul_params(self):
        """Warn when a dense ``dim=2`` parameter is used inside a
        ``Mat_Mul``. Such parameters fall back to the slower
        scipy/ndarray fancy-indexing path and do not benefit from the
        vectorised mutable-matrix Jacobian code. Each offending param is
        warned about exactly once per system.
        """
        warned = set()
        for eqn_name, eqn in self.EQNs.items():
            if not eqn.mixed_matrix_vector:
                continue
            for mm in eqn.RHS.atoms(Mat_Mul):
                for arg in mm.args:
                    free_paras = {s for s in arg.free_symbols if isinstance(s, Para)}
                    for sym in free_paras:
                        if sym.name not in self.PARAM:
                            continue
                        p = self.PARAM[sym.name]
                        if p.dim == 2 and not p.sparse and sym.name not in warned:
                            warnings.warn(
                                f"Parameter {sym.name!r} is a dense 2-D "
                                f"``Param(..., dim=2, sparse=False)`` used "
                                f"inside ``Mat_Mul``. Dense matrices bypass "
                                f"the vectorised mutable-matrix Jacobian "
                                f"fast path and fall back to a slower "
                                f"scipy/ndarray indexing path. Declare "
                                f"{sym.name!r} with ``sparse=True`` for "
                                f"substantially better performance.",
                                UserWarning, stacklevel=3
                            )
                            warned.add(sym.name)

    def FormJac(self,
                y
                ):
        self.assign_eqn_var_address(y)

        self._check_no_timevar_sparse_matrices()
        self._warn_dense_matmul_params()

        Fy_list = self.Fy(y, self.a.object_list, self.var_address.object_list)

        for fy in Fy_list:
            EqnName = fy[0]
            EqnAddr = self.a[EqnName]
            VarName = fy[1]
            VarAddr = self.var_address[VarName]
            DiffVar = fy[2].diff_var
            DeriExpr = fy[2].RHS

            DiffVarEqn = Eqn('DiffVarEqn' + DiffVar.name, DiffVar)
            args = self.obtain_eqn_args(DiffVarEqn, y, 0)
            DiffVarValue = Array(DiffVarEqn.NUM_EQN(*args), dim=1)

            # Mutable-matrix-block detection. This predicate MUST match
            # ``JacBlock.is_mutable_matrix`` exactly — FormJac picks
            # Value0 here and JacBlock picks the downstream codegen path
            # from the *same* flag, so any divergence between the two
            # locations would produce blocks whose Value0 sparsity
            # pattern disagreed with the kernel the code generator then
            # emitted (e.g. a flat-start ``Diag(x)`` Value0 would
            # collapse to empty while the scatter-add loop expected a
            # full diagonal). Both locations call
            # :func:`is_constant_matrix_deri` from ``jac.py`` to keep
            # the predicate single-sourced.
            #
            # Criterion: the derivative is matrix-valued AND its
            # symbolic form depends on at least one state variable. We
            # probe "matrix-valued" from the already-evaluated
            # ``fy[3]`` (either a ``csc_array`` from a sparse
            # expression or an ndarray with ``ndim == 2`` from a dense
            # one).
            fy_value = fy[3]
            fy_is_matrix = (
                isinstance(fy_value, csc_array)
                or (isinstance(fy_value, np.ndarray) and fy_value.ndim == 2)
            )
            is_mutable_matrix_block = (
                fy_is_matrix and not is_constant_matrix_deri(DeriExpr)
            )
            # For such blocks we MUST use sps.diags (not np.diagflat) so
            # that Value0's sparsity pattern captures ALL structural
            # non-zeros (union of each term's pattern), independently of
            # the numerical values at y0. This guarantees
            # Value0.tocsc().data order == runtime .tocsc().data order,
            # so the runtime can do O(nnz) direct data copy without any
            # fancy indexing.
            if is_mutable_matrix_block:
                sparse_expr = DeriExpr.replace(Diag, SpDiag)
                sparse_eqn = Eqn('_MutMatJb_' + EqnName + '_' + DiffVar.name,
                                 sparse_expr)
                sparse_args = self.obtain_eqn_args(sparse_eqn, y, 0)
                # Perturb variable values with distinct non-zero samples so
                # that ``sps.diags(v) @ M`` doesn't collapse to the empty
                # matrix when v happens to be zero at y0 (as in flat start).
                # Value0 is only used for sparsity-pattern tracking — the
                # data is overwritten at the first J_() call — so the
                # perturbation is harmless.
                rng = np.random.default_rng(seed=20260412)
                perturbed_args = []
                for symbol, arg in zip(sparse_eqn.SYMBOLS.values(), sparse_args):
                    if symbol.name in y.var_list:
                        perturbed_args.append(rng.random(arg.shape) + 1.0)
                    else:
                        perturbed_args.append(arg)
                Value0 = sparse_eqn.NUM_EQN(*perturbed_args)
                if not isinstance(Value0, csc_array):
                    Value0 = csc_array(Value0)
            else:
                # The value of deri can be either matrix, vector, or scalar.
                if isinstance(fy[3], csc_array):
                    Value0 = fy[3]
                else:
                    Value0 = np.array(fy[3])

            jb = JacBlock(EqnName,
                          EqnAddr,
                          DiffVar,
                          DiffVarValue,
                          VarAddr,
                          DeriExpr,
                          Value0)
            self.jac.add_block(EqnName, DiffVar, jb)

        self.jac.shape = np.array([self.eqn_size, self.vsize], dtype=int)
        self.jac.coordinate0 = np.ndarray([0, 0], dtype=int)

    def FormPartialJac(self,
                       y,
                       eqn_list: List[str] = None,
                       var_list: List[str] = None):

        def is_sublist(sub, main):
            n = len(sub)
            for i in range(len(main) - n + 1):
                if main[i:i + n] == sub:
                    return True
            return False

        if eqn_list is not None:
            non_existent_eqn = [eqn for eqn in eqn_list if eqn not in self.a.object_list]
            if len(non_existent_eqn) > 0:
                raise ValueError(f"Non-existent equation: {non_existent_eqn}")
            if not is_sublist(eqn_list, self.a.object_list):
                raise ValueError(f"Given equation list is discontinuous, which cannot form sub-Jacobian.")
        else:
            eqn_list = deepcopy(self.a.object_list)

        if var_list is not None:
            non_existent_var = [var for var in var_list if var not in self.var_address.object_list]
            if len(non_existent_var) > 0:
                raise ValueError(f"Non-existent variable: {non_existent_var}")
            if not is_sublist(var_list, self.var_address.object_list):
                raise ValueError(f"Given variable list is discontinuous, which cannot form sub-Jacobian.")
        else:
            var_list = deepcopy(self.var_address.object_list)

        jac = Jac()

        Fy_list = self.Fy(y, eqn_list, var_list)

        for fy in Fy_list:
            EqnName = fy[0]
            EqnAddr = self.a[EqnName]
            VarName = fy[1]
            VarAddr = self.var_address[VarName]
            DiffVar = fy[2].diff_var
            DeriExpr = fy[2].RHS

            DiffVarEqn = Eqn('DiffVarEqn' + DiffVar.name, DiffVar)
            args = self.obtain_eqn_args(DiffVarEqn, y, 0)
            DiffVarValue = Array(DiffVarEqn.NUM_EQN(*args), dim=1)

            # The value of deri can be either matrix, vector, or scalar(number). We cannot reshape it.
            if isinstance(fy[3], csc_array):
                Value0 = fy[3]
            else:
                Value0 = np.array(fy[3])

            jb = JacBlock(EqnName,
                          EqnAddr,
                          DiffVar,
                          DiffVarValue,
                          VarAddr,
                          DeriExpr,
                          Value0)
            jac.add_block(EqnName, DiffVar, jb)

        eqn_start = self.a[eqn_list[0]].start
        eqn_end = self.a[eqn_list[-1]].stop
        var_start = self.var_address[var_list[0]].start
        var_end = self.var_address[var_list[-1]].stop

        jac.shape = np.array([eqn_end - eqn_start, var_end - var_start], dtype=int)
        jac.coordinate0 = np.array([eqn_start, var_start], dtype=int)

        return jac

    @property
    def eqn_size(self):
        # total size of all the equations
        return np.sum(np.array(list(self.esize.values())))

    def is_param_defined(self, param: str) -> bool:

        if param in self.SYMBOLS:
            return True
        else:
            return False

    with warnings.catch_warnings():
        warnings.simplefilter("once")

    def param_initializer(self, name, param: ParamBase):
        if not self.is_param_defined(name):
            warnings.warn(f'Parameter {name} not defined in equations!')
        if isinstance(param, ParamBase):
            self.PARAM[name] = param
            if param.triggerable:
                self.triggerable_quantity[param.name] = param.trigger_var
        else:
            raise TypeError(f"Unsupported parameter type {type(param)}")

    def update_param(self, *args):

        if isinstance(args[0], str):
            # Update specified params
            param: str = args[0]
            value: Union[np.ndarray, list, Number] = args[1]
            try:
                if param in self.PARAM:
                    self.PARAM[param].v = value
            except KeyError:
                warnings.warn(f'Equations have no parameter: {param}')
        elif isinstance(args[0], Vars):
            # Update params with Vars. For example, to update x0 in trapezoid rules.
            vars_: Vars = args[0]
            for param_name in self.PARAM.keys():
                if param_name in vars_.var_list:
                    self.PARAM[param_name].v = vars_[param_name]

    def eval(self, eqn_name: str, *args: Union[np.ndarray]) -> np.ndarray:
        """
        Evaluate equations
        :param eqn_name:
        :param args:
        :return:
        """
        return Array(self.EQNs[eqn_name].NUM_EQN(*args), dim=1)

    def trigger_param_updater(self, y):
        # update/initialize triggerable params
        for para_name, trigger_var in self.triggerable_quantity.items():
            trigger_func = self.PARAM[para_name].trigger_fun
            args = []
            for var in trigger_var:
                var_value = None
                if var in self.PARAM:
                    var_value = self.PARAM[var].v
                else:
                    if var in y.var_list:
                        var_value = y[var]
                if var_value is None:
                    raise ValueError(f'Para/iVar {var} not defined')
                else:
                    args.append(var_value)
            temp = trigger_func(*args)
            if self.PARAM[para_name].v is None:
                self.PARAM[para_name].v = temp
            else:
                if type(temp) is not type(self.PARAM[para_name].v):
                    raise TypeError(
                        f"The return types of trigger func for param {para_name} must be {type(self.PARAM[para_name].v)}")

    def obtain_eqn_args(self, eqn: Eqn, y: Vars, t=0) -> List[np.ndarray]:
        """
        Obtain the args of equations
        """

        self.trigger_param_updater(y)

        args = []
        for symbol in eqn.SYMBOLS.values():
            value_obtained = False
            if symbol.name in self.PARAM:
                temp = self.PARAM[symbol.name].get_v_t(t)
                if temp is None:
                    raise TypeError(f'Parameter {symbol.name} uninitialized')
                args.append(temp)
                value_obtained = True
            else:
                if symbol.name in y.var_list:
                    args.append(y[symbol.name])
                    value_obtained = True
            if not value_obtained:
                raise ValueError(f'Cannot find the values of variable {symbol.name}')
        return args

    def eval_diffs(self, eqn_name: str, var_name: str, *args: np.ndarray) -> np.ndarray:
        """
        Evaluate derivative of equations
        :param eqn_name:
        :param var_name:
        :param args:
        :return:
        """
        return self.EQNs[eqn_name].derivatives[var_name].NUM_EQN(*args)

    def evalf(self, *args) -> np.ndarray:
        pass


class AE(Equations):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 matrix_container='scipy'):
        super().__init__(eqn, name, matrix_container)

        # Check if some equation in self.eqn is Eqn.
        # If not, raise error
        if len(self.f_list) > 0:
            raise ValueError(f'Ode found. This object should be DAE!')

    def assign_eqn_var_address(self, y: Vars):
        """
        ASSIGN ADDRESSES TO EQUATIONS
        """

        temp = 0
        for eqn_name in self.EQNs.keys():
            geval = self.g(y, eqn_name)
            if isinstance(geval, Number):
                eqn_size = 1
            else:
                eqn_size = geval.shape[0]
            self.a.update(eqn_name, eqn_size)
            temp = temp + eqn_size
            self.esize[eqn_name] = eqn_size
        self.var_address = y.a
        self.vsize = y.total_size

    def g(self, y: Vars, eqn: str = None) -> np.ndarray:
        """

        :param y:
        :param eqn:
        :return:
        """
        temp = []
        if not eqn:
            for eqn_name, eqn_ in self.EQNs.items():
                args = self.obtain_eqn_args(eqn_, y)
                g_eqny = self.eval(eqn_name, *args)
                g_eqny = g_eqny.toarray() if isinstance(g_eqny, csc_array) else g_eqny
                temp.append(g_eqny.reshape(-1, ))
            return np.hstack(temp)
        else:
            args = self.obtain_eqn_args(self.EQNs[eqn], y)
            return self.eval(eqn, *args)

    def gy(self,
           y: Vars,
           eqn_list: List[str] = None,
           var_list: List[str] = None) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
        """
        generate Jacobian matrices of Eqn object with respect to var object
        :param y:
        :param eqn_list:
        :param var_list:
        :return: List[Tuple[Equation_name, var_name, np.ndarray]]
        """
        if not eqn_list:
            eqn_list = list(self.EQNs.keys())
        if not var_list:
            var_list = list(y.var_list)

        gy: List[Tuple[str, str, EqnDiff, np.ndarray]] = []

        for eqn_name in eqn_list:
            eqn_diffs: Dict[str, EqnDiff] = self.EQNs[eqn_name].derivatives
            for var_name in var_list:
                for key, value in eqn_diffs.items():
                    if var_name == value.diff_var_name:  # f is viewed as f[k]
                        args = self.obtain_eqn_args(eqn_diffs[key], y)
                        temp = self.eval_diffs(eqn_name, key, *args)
                        gy = [*gy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return gy

    def Fy(self,
           y,
           eqn_list: List[str] = None,
           var_list: List[str] = None):
        return self.gy(y, eqn_list, var_list)

    def evalf(self, expr: Expr, y: Vars) -> np.ndarray:
        eqn = Eqn('Solverz evalf temporal equation', expr)
        args = self.obtain_eqn_args(eqn, y)
        return eqn.NUM_EQN(*args)

    def __repr__(self):
        if not self.eqn_size:
            return f"Algebraic equation {self.name} with addresses uninitialized"
        else:
            return f"Algebraic equation {self.name} ({self.eqn_size}×{self.vsize})"


class FDAE(AE):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 nstep: int,
                 name: str = None,
                 matrix_container='scipy'):
        super().__init__(eqn, name, matrix_container)

        self.nstep = nstep

    def __repr__(self):
        if not self.eqn_size:
            return f"FDAE {self.name} with addresses uninitialized"
        else:
            return f"FDAE {self.name} ({self.eqn_size}×{self.vsize})"


class DAE(Equations):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 matrix_container='scipy'
                 ):

        super().__init__(eqn, name, matrix_container)

        self.state_num: int = 0  # number of state variables
        self.algebra_num: int = 0  # number of algebraic variables

        # Check if some equation in self.eqn is Ode.
        # If not, raise error
        if len(self.f_list) == 0:
            raise ValueError(f'No ODE found. You should initialise AE instead!')

    def evalf(self, expr: Expr, t, y: Vars) -> np.ndarray:
        eqn = Eqn('Solverz evalf temporary equation', expr)
        args = self.obtain_eqn_args(eqn, y, t)
        return eqn.NUM_EQN(*args)

    def assign_eqn_var_address(self, y: Vars):
        """
        ASSIGN ADDRESSES TO EQUATIONS f and g
        """

        self.state_num = 0
        self.algebra_num = 0
        for eqn_name in self.f_list + self.g_list:
            if eqn_name in self.f_list:
                feval = self.f(None, y, eqn=eqn_name)
                lhs_eval = self.eval_lhs(None, y, eqn=eqn_name)
                if isinstance(feval, Number):
                    rhs_size = 1
                else:
                    rhs_size = feval.shape[0]
                if isinstance(lhs_eval, Number):
                    lhs_size = 1
                else:
                    lhs_size = lhs_eval.shape[0]
                eqn_size = np.max([rhs_size, lhs_size])
                self.state_num += eqn_size
            elif eqn_name in self.g_list:
                geval = self.g(None, y, eqn=eqn_name)
                if np.max(np.abs(geval)) > 1e-5:
                    warnings.warn(
                        f'Inconsistent initial values for algebraic equation: {eqn_name}, with deviation {np.max(np.abs(geval))}!')
                if isinstance(geval, Number):
                    eqn_size = 1
                else:
                    eqn_size = geval.shape[0]
                self.algebra_num += eqn_size
            else:
                raise ValueError(f'Non-existent equation: {eqn_name}')
            self.a.update(eqn_name, eqn_size)
            self.esize[eqn_name] = eqn_size

        self.var_address = y.a
        self.vsize = self.var_address.total_size

    def F(self, t, y) -> np.ndarray:
        """
        Return [f(t,x,y), g(t,x,y)]
        :param t: time
        :param y: Vars
        :return:
        """
        if len(self.g_list) > 0:
            return np.concatenate([self.f(t, y), self.g(t, y)])
        else:
            return self.f(t, y)

    def f(self, t, y, eqn=None) -> np.ndarray:

        temp = []
        if eqn:
            if eqn in self.f_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], y, t)
                temp.append(self.eval(eqn, *args).reshape(-1, ))
        else:
            for eqn in self.f_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], y, t)
                temp.append(self.eval(eqn, *args).reshape(-1, ))

        return np.hstack(temp)

    def eval_lhs(self, t, y, eqn=None) -> np.ndarray:

        temp = []
        if eqn:
            if eqn in self.f_list:
                ode = self.EQNs[eqn]
                if isinstance(ode, Ode):
                    lhs_eqn = Eqn('lhs_' + eqn, ode.diff_var)
                    args = self.obtain_eqn_args(lhs_eqn, y, t)
                    temp.append(Array(lhs_eqn.NUM_EQN(*args), dim=1))
                else:
                    raise TypeError(f"Equation {ode.name} in f_list is not Ode!")
        else:
            for eqn in self.f_list:
                ode = self.EQNs[eqn]
                if isinstance(ode, Ode):
                    lhs_eqn = Eqn('lhs_' + eqn, ode.diff_var)
                    args = self.obtain_eqn_args(lhs_eqn, y, t)
                    temp.append(Array(lhs_eqn.NUM_EQN(*args), dim=1))
                else:
                    raise TypeError(f"Equation {ode.name} in f_list is not Ode!")

        return np.hstack(temp)

    def g(self, t, y, eqn=None) -> np.ndarray:
        """

        `xys` is either:
          - two arguments, e.g. state vars x, and numerical equation y
          - one argument, e.g. state vars y.

        """

        if len(self.g_list) == 0:
            raise ValueError(f'No AE found in {self.name}!')

        temp = []
        if eqn:
            if eqn in self.g_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], y, t)
                temp.append(self.eval(eqn, *args).reshape(-1, ))
        else:
            for eqn in self.g_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], y, t)
                temp.append(self.eval(eqn, *args).reshape(-1, ))

        return np.hstack(temp)

    def fy(self,
           t,
           y: Vars,
           eqn_list: List[str] = None,
           var_list: List[str] = None
           ) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
        """
        generate partial derivatives of f w.r.t. y
        """

        if not eqn_list:
            eqn_list = list(self.f_list)
        else:
            eqn_list = list(set(eqn_list) & set(self.f_list))

        if not var_list:
            var_list = list(y.var_list)

        fy: List[Tuple[str, str, EqnDiff, np.ndarray]] = []

        for eqn_name in eqn_list:
            eqn_diffs: Dict[str, EqnDiff] = self.EQNs[eqn_name].derivatives
            for var_name in var_list:
                for key, value in eqn_diffs.items():
                    if var_name == value.diff_var_name:
                        args = self.obtain_eqn_args(eqn_diffs[key], y, t)
                        temp = self.eval_diffs(eqn_name, key, *args)
                        fy = [*fy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return fy

    def gy(self,
           t,
           y: Vars,
           eqn_list: List[str] = None,
           var_list: List[str] = None
           ) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
        """
        generate partial derivatives of g w.r.t. y
        """

        if len(self.g_list) == 0:
            raise ValueError(f'No AE found in {self.name}!')

        if not eqn_list:
            eqn_list = list(self.g_list)
        else:
            eqn_list = list(set(eqn_list) & set(self.g_list))

        if not var_list:
            var_list = list(y.var_list)

        gy: List[Tuple[str, str, EqnDiff, np.ndarray]] = []

        for eqn_name in eqn_list:
            eqn_diffs: Dict[str, EqnDiff] = self.EQNs[eqn_name].derivatives
            for var_name in var_list:
                for key, value in eqn_diffs.items():
                    if var_name == value.diff_var_name:
                        args = self.obtain_eqn_args(eqn_diffs[key], y, t)
                        temp = self.eval_diffs(eqn_name, key, *args)
                        gy = [*gy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return gy

    def Fy(self,
           y,
           eqn_list: List[str] = None,
           var_list: List[str] = None
           ) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
        fg_xy = self.fy(0, y, eqn_list, var_list)
        if len(self.g_list) > 0:
            fg_xy.extend(self.gy(0, y, eqn_list, var_list))
        return fg_xy

    @property
    def M(self):
        """
        return the singular mass matrix, M, of dae
        Row of 1 in M corresponds to the differential equation
        Col of 1 in M corresponds to the state variable
        """
        if self.state_num == 0:
            raise ValueError("DAE address uninitialized!")

        row = []
        col = []
        for eqn_name in self.f_list:
            eqn = self.EQNs[eqn_name]
            equation_address = self.a.v[eqn_name]
            if isinstance(eqn, Ode):
                diff_var = eqn.diff_var
                if isinstance(diff_var, iVar):
                    variable_address = self.var_address.v[diff_var.name]
                elif isinstance(diff_var, IdxVar):
                    var_idx = diff_var.index
                    var_name = diff_var.name0
                    if is_integer(var_idx):
                        variable_address = self.var_address.v[var_name][var_idx: var_idx + 1]
                    elif isinstance(var_idx, str):
                        variable_address = self.var_address.v[var_name][np.ix_(self.PARAM[var_idx].v.reshape((-1,)))]
                    elif isinstance(var_idx, slice):
                        variable_address = self.var_address.v[var_name][var_idx]
                    elif isinstance(var_idx, Expr):
                        raise TypeError(f"Index of {diff_var} cannot be sympy.Expr!")
                    elif isinstance(var_idx, list):
                        variable_address = self.var_address.v[var_name][var_idx]
                    else:
                        raise TypeError(f"Unsupported variable index {var_idx} in equation {eqn_name}")
                else:
                    raise NotImplementedError
                eqn_address_list = equation_address.tolist()
                var_address_list = variable_address.tolist()
                if len(eqn_address_list) != len(var_address_list):
                    raise ValueError(
                        f"Incompatible eqn address length {len(eqn_address_list)} and variable address length {len(var_address_list)}")
                row.extend(eqn_address_list)
                col.extend(var_address_list)
            else:
                raise ValueError("Equation in f_list is non-Ode.")

        if self.matrix_container == 'scipy':
            return csc_array((np.ones((self.state_num,)), (row, col)), (self.eqn_size, self.vsize))
        elif self.matrix_container == 'cvxopt':
            raise NotImplementedError("Not implemented!")
        else:
            raise ValueError(f"Unsupported matrix container {self.matrix_container}")

    @property
    def alg_eqn_addr(self):
        addr_list = []
        for eqn in self.g_list:
            addr_list.extend(self.a.v[eqn].tolist())

        return addr_list

    def __repr__(self):
        if not self.eqn_size:
            return f"DAE {self.name} with addresses uninitialized"
        else:
            return f"DAE {self.name} ({self.eqn_size}×{self.vsize})"
