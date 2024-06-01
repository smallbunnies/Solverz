from __future__ import annotations

from typing import Union, List, Dict, Callable

import numpy as np
from sympy import Symbol, Expr, latex, Derivative, sympify
from sympy import lambdify as splambdify
from sympy.abc import t, x

from Solverz.sym_algebra.symbols import iVar, Para, IdxVar, idx, IdxPara, iAliasVar, IdxAliasVar
from Solverz.variable.ssymbol import Var
from Solverz.sym_algebra.functions import Mat_Mul, Slice, F
from Solverz.sym_algebra.matrix_calculus import MixedEquationDiff
from Solverz.sym_algebra.transform import finite_difference, semi_descritize
from Solverz.num_api.custom_function import numerical_interface
from Solverz.variable.ssymbol import sSym2Sym


def sVar2Var(var: Union[Var, iVar, List[iVar, Var]]) -> Union[iVar, List[iVar]]:
    if isinstance(var, list):
        return [arg.symbol if isinstance(arg, Var) else arg for arg in var]
    else:
        return var.symbol if isinstance(var, Var) else var


class Eqn:
    """
    The Equation object
    """

    def __init__(self,
                 name: str,
                 eqn):
        if not isinstance(name, str):
            raise ValueError("Equation name must be string!")
        self.name: str = name
        self.LHS = 0
        self.RHS = sympify(eqn)
        self.SYMBOLS: Dict[str, Symbol] = self.obtain_symbols()

        # if the eqn has Mat_Mul, then label it as mixed-matrix-vector equation
        if self.expr.has(Mat_Mul):
            self.mixed_matrix_vector = True
        else:
            self.mixed_matrix_vector = False

        self.NUM_EQN: Callable = self.lambdify()
        self.derivatives: Dict[str, EqnDiff] = dict()

    def obtain_symbols(self) -> Dict[str, Symbol]:
        temp_dict = dict()
        for symbol_ in list((self.LHS - self.RHS).free_symbols):
            if isinstance(symbol_, (iVar, Para, idx, iAliasVar)):
                temp_dict[symbol_.name] = symbol_
            elif isinstance(symbol_, (IdxVar, IdxPara, IdxAliasVar)):
                temp_dict[symbol_.name0] = symbol_.symbol0
                temp_dict.update(symbol_.SymInIndex)

        # to sort in lexicographic order
        sorted_dict = {key: temp_dict[key] for key in sorted(temp_dict)}
        return sorted_dict

    def lambdify(self) -> Callable:
        return splambdify(self.SYMBOLS.values(), self.RHS, [numerical_interface, 'numpy'])

    def eval(self, *args: Union[np.ndarray]) -> np.ndarray:
        return self.NUM_EQN(*args)

    def derive_derivative(self):
        """"""
        for symbol_ in list(self.RHS.free_symbols):
            # differentiate only to variables
            if isinstance(symbol_, IdxVar):  # if the equation contains Indexed variables
                idx_ = symbol_.index
                if self.mixed_matrix_vector:
                    diff = MixedEquationDiff(self.RHS, symbol_)
                else:
                    diff = self.RHS.diff(symbol_)
                self.derivatives[symbol_.name] = EqnDiff(name=f'Diff {self.name} w.r.t. {symbol_.name}',
                                                         eqn=diff,
                                                         diff_var=symbol_,
                                                         var_idx=idx_.name if isinstance(idx_, idx) else idx_)
            elif isinstance(symbol_, iVar):
                if self.mixed_matrix_vector:
                    diff = MixedEquationDiff(self.RHS, symbol_)
                else:
                    diff = self.RHS.diff(symbol_)
                self.derivatives[symbol_.name] = EqnDiff(name=f'Diff {self.name} w.r.t. {symbol_.name}',
                                                         eqn=diff,
                                                         diff_var=symbol_)

    @property
    def expr(self):
        return self.LHS - self.RHS

    def subs(self, *args, **kwargs):
        return self.RHS.subs(*args, **kwargs)

    def __repr__(self):
        # sympy objects' printing prefers __str__() to __repr__()
        return self.LHS.__str__() + r"=" + self.RHS.__str__()

    def _repr_latex_(self):
        """
        So that jupyter notebook can display latex equation of Eqn object.
        :return:
        """
        return r"$\displaystyle %s$" % (latex(self.LHS) + r"=" + latex(self.RHS))


class EqnDiff(Eqn):
    """
    To store the derivatives of equations W.R.T. variables
    """

    def __init__(self, name: str, eqn: Expr, diff_var: Symbol, var_idx=None):
        super().__init__(name, eqn)
        self.diff_var = diff_var
        self.diff_var_name = diff_var.name0 if isinstance(diff_var, IdxVar) else diff_var.name
        self.var_idx = var_idx  # df/dPi[i] then var_idx=i
        self.var_idx_func = None
        if self.var_idx is not None:
            if isinstance(self.var_idx, slice):
                temp = []
                if var_idx.start is not None:
                    temp.append(var_idx.start)
                if var_idx.stop is not None:
                    temp.append(var_idx.stop)
                if var_idx.step is not None:
                    temp.append(var_idx.step)
                self.var_idx_func = Eqn('To evaluate var_idx of variable' + self.diff_var.name, Slice(*temp))
            elif isinstance(self.var_idx, Expr):
                self.var_idx_func = Eqn('To evaluate var_idx of variable' + self.diff_var.name, self.var_idx)
        self.LHS = Derivative(F, diff_var)
        self.dim = -1
        self.v_type = ''


class Ode(Eqn):
    r"""
    The class for ODE reading

    .. math::

         \frac{\mathrm{d}y}{\mathrm{d}t}=f(t,y)

    where $y$ is the state vector.

    """

    def __init__(self, name: str,
                 f,
                 diff_var: Union[iVar, IdxVar, Var]):
        super().__init__(name, sSym2Sym(f))
        diff_var = sSym2Sym(diff_var)
        self.diff_var = diff_var
        self.LHS = Derivative(diff_var, t)


class Pde(Eqn):
    """
    The class of partial differential equations
    """
    pass


class HyperbolicPde(Pde):
    r"""
    The class for hyperbolic PDE reading

    .. math::

         \frac{\partial{u}}{\partial{t}}+\frac{\partial{f(u)}}{\partial{x}}=S(u)

    where $u$ is the state vector, $f(u)$ is the flux function and $S(u)$ is the source term.

    Parameters
    ==========

    two_dim_var : iVar or list of Var

        Specify the two-dimensional variables in the PDE. Some of the variables, for example, the mass flow $\dot{m}$ in
        the heat transmission equation, are not two-dimensional variables.

    """

    def __init__(self, name: str,
                 diff_var: iVar | Var,
                 flux: Expr = 0,
                 source: Expr = 0,
                 two_dim_var: Union[iVar, Var, List[iVar | Var]] = None):
        if isinstance(source, (float, int)):
            source = sympify(source)
        super().__init__(name, source)
        diff_var = sVar2Var(diff_var)
        two_dim_var = sVar2Var(two_dim_var) if two_dim_var is not None else None
        self.diff_var = diff_var
        if isinstance(flux, (float, int)):
            flux = sympify(flux)
        if isinstance(source, (float, int)):
            flux = sympify(flux)
        self.flux = flux
        self.source = source
        self.two_dim_var = [two_dim_var] if isinstance(two_dim_var, iVar) else two_dim_var
        self.LHS = Derivative(diff_var, t) + Derivative(flux, x)

    def derive_derivative(self):
        pass

    def finite_difference(self, scheme='central diff', direction=None, M: int = 0, dx=None):
        r"""
        Discretize hyperbolic PDE as AEs.

        Parameters
        ==========

        scheme : str

            1 - Central difference

            .. math::

                \frac{\partial{u}}{\partial{t}}\approx\frac{u_{i+1}^{j+1}-u_{i+1}^{j}+u_{i}^{j+1}-u_{i}^{j}}{2\Delta t}

            .. math::

                \frac{\partial{f(u)}}{\partial{x}}\approx\frac{f(u_{i+1}^{j+1})-f(u_{i}^{j+1})+f(u_{i+1}^{j})-f(u_{i}^{j})}{2\Delta x}

            .. math::

                S(u)\approx S\left(\frac{u_{i+1}^{j+1}+u_{i}^{j+1}+u_{i+1}^{j}+u_{i}^{j}}{4}\right)

            2 - Backward Time Backward/Forward Space

            If direction equals 1, then do backward space difference, which derives

            .. math::

                \frac{\partial{u}}{\partial{t}}\approx\frac{u_{i+1}^{j+1}-u_{i+1}^{j}}{\Delta t}

            .. math::

                \frac{\partial{f(u)}}{\partial{x}}\approx\frac{f(u_{i+1}^{j+1})-f(u_{i}^{j+1})}{\Delta x}

            .. math::

                S(u)\approx S\left(u_{i+1}^{j+1}\right)

            If direction equals -1, then do forward space difference, which derives

            .. math::

                \frac{\partial{u}}{\partial{t}}\approx\frac{u_{i}^{j+1}-u_{i}^{j}}{\Delta t}

            .. math::

                \frac{\partial{f(u)}}{\partial{x}}\approx\frac{f(u_{i+1}^{j+1})-f(u_{i}^{j+1})}{\Delta x}

            .. math::

                S(u)\approx S\left(u_{i}^{j+1}\right)

        direction : int

            To tell which side of boundary conditions is given in scheme 2.

        M : int

        The total number of spatial sections.

        dx : Number

        Spatial difference step size

        Returns
        =======

        AE : Eqn

            Let's take central difference as an example, this function returns the algebraic equation

            .. math::

                \begin{aligned}
                    0=&\Delta x(\tilde{u}[1:M]-\tilde{u}^0[1:M]+\tilde{u}[0:M-1]-\tilde{u}^0[0:M-1])+\\
                      &\Delta t(f(\tilde{u}[1:M])-f(\tilde{u}[0:M-1])+f(\tilde{u}^0[1:M])-f(\tilde{u}^0[0:M-1]))+\\
                      &2\Delta x\Delta t\cdot S\left(\tilde{u}[1:M]-\tilde{u}^0[1:M]+\tilde{u}[0:M-1]-\tilde{u}^0[0:M-1]}{4}\right)
                \end{aligned}

            where we denote by vector $\tilde{u}$ the discrete spatial distribution of state $u$, by $\tilde{u}^0$ the
            initial value of $\tilde{u}$, and by $M$ the last index of $\tilde{u}$.

        """
        if isinstance(M, (int, np.integer)):
            if M < 0:
                raise ValueError(f'Total nunmber of PDE sections {M} < 0')
        else:
            raise TypeError(f'Do not support M of type {type(M)}')
        if M == 0:
            M = idx('M')

        if scheme == 'central diff':
            return Eqn('FDM of ' + self.name + 'w.r.t.' + self.diff_var.name + 'using central diff',
                       finite_difference(self.diff_var,
                                         self.flux,
                                         self.source,
                                         self.two_dim_var,
                                         M,
                                         'central diff',
                                         dx=dx))
        elif scheme == 'euler':
            return Eqn('FDM of ' + self.name + 'w.r.t.' + self.diff_var.name + 'using Euler',
                       finite_difference(self.diff_var,
                                         self.flux,
                                         self.source,
                                         self.two_dim_var,
                                         M,
                                         'euler',
                                         direction=direction,
                                         dx=dx))

    def semi_discretize(self,
                        a0=None,
                        a1=None,
                        scheme='TVD1',
                        M: int = 0,
                        output_boundary=True,
                        dx=None) -> List[Eqn]:
        r"""
        Semi-discretize the hyperbolic PDE of nonlinear conservation law as ODEs using the Kurganov-Tadmor scheme
        (see [Kurganov2000]_). The difference stencil is as follows, with $x_{j+1}-x_{j}=\Delta x$.

            .. image:: ../../pics/difference_stencil.png
               :height: 100

        Parameters
        ==========

        a0 : Expr

            Maximum local speed $a_{j+1/2}$, with formula

            .. math::

                a_{j+1/2}=\max\qty{\rho\qty(\pdv{f}{u}\qty(u^+_{j+1/2})),\rho\qty(\pdv{f}{u}\qty(u^-_{j+1/2}))},

            where

            .. math::

                \rho(A)=\max_i|\lambda_i(A)|.

            If $a_0$ or $a_1$ is None, then they will be set as ``Para`` ``ajp12`` and ``ajm12`` respectively.  

        a1 : Expr

            Maximum local speed $a_{j-1/2}$, with formula

            .. math::

                a_{j-1/2}=\max\qty{\rho\qty(\pdv{f}{u}\qty(u^+_{j-1/2})),\rho\qty(\pdv{f}{u}\qty(u^-_{j-1/2}))}.

        scheme : str

            If scheme==1, 2nd scheme else, else, use 1st scheme.

        M : int

            The total number of spatial sections.

        output_boundary : bool

            If true, output equations about the boundary conditions. For example,

           >>> from Solverz import HyperbolicPde, iVar
           >>> T = iVar('T')
           >>> p = HyperbolicPde(name = 'heat transfer', diff_var=T, flux=T)
           >>> p.semi_discretize(a0=1,a2=1, scheme=2, M=2, output_boundary=True)
           1
           >>> p.semi_discretize(a0=1,a2=1, scheme=2, M=2, output_boundary=False)
           2

        dx : Number

            spatial difference step size

        Returns
        =======

        ODE : List[Union[Ode, Eqn]]

            This function returns the for $2\leq j\leq M-2$

            .. math::

                \dv{t}u_j=-\frac{H_{j+1/2}-H_{j-1/2}}{\Delta x}+S(u_j)

            and for $j=1,M-1$

            .. math::

                \dv{t}u_j=-\frac{f(u_{j+1})-f(u_{j-1})}{2\Delta x}+\frac{a_{j+1/2}(u_{j+1}-u_j)-a_{j-1/2}(u_j-u_{j-1})}{2\Delta x}+S(u_j),

            where

            .. math::

                H_{j+1/2}=\frac{f(u^+_{j+1/2})+f(u^-_{j+1/2})}{2}-\frac{a_{j+1/2}}{2}\qty[u^+_{j+1/2}-u^-_{j+1/2}],

            .. math::

                H_{j-1/2}=\frac{f(u^+_{j-1/2})+f(u^-_{j-1/2})}{2}-\frac{a_{j-1/2}}{2}\qty[u^+_{j-1/2}-u^-_{j-1/2}],

            .. math::

                u^+_{j+1/2}=u_{j+1}-\frac{\Delta x}{2}(u_x)_{j+1},\quad u^-_{j+1/2}=u_j+\frac{\Delta x}{2}(u_x)_j,

            .. math::

                u^+_{j-1/2}=u_{j}-\frac{\Delta x}{2}(u_x)_{j},\quad u^-_{j-1/2}=u_{j-1}+\frac{\Delta x}{2}(u_x)_{j-1},

            .. math::

                (u_x)_j=\operatorname{minmod}\qty(\theta\frac{u_j-u_{j-1}}{\Delta x},\frac{u_{j+1}-u_{j-1}}{2\Delta x},\theta\frac{u_{j+1}-u_{j}}{\Delta x}),\quad \theta\in[1,2],

            and by linear extrapolation

            .. math::

                u_0=2u_\text{L}-u_1,\quad u_M=2u_\text{R}-u_{M-1}.


        .. [Kurganov2000] Alexander Kurganov, Eitan Tadmor, New High-Resolution Central Schemes for Nonlinear Conservation Laws and Convection–Diffusion Equations, Journal of Computational Physics, Volume 160, Issue 1, 2000, Pages 241-282, `<https://doi.org/10.1006/jcph.2000.6459>`_

        """
        if isinstance(M, (int, np.integer)):
            if M < 0:
                raise ValueError(f'Total nunmber of PDE sections {M} < 0')
        else:
            raise TypeError(f'Do not support M of type {type(M)}')
        if M == 0:
            M = idx('M')
        u = self.diff_var
        dae_list = []
        if scheme == 'TVD2':
            eqn_dict = semi_descritize(self.diff_var,
                                       self.flux,
                                       self.source,
                                       self.two_dim_var,
                                       M,
                                       scheme='TVD2',
                                       a0=a0,
                                       a1=a1,
                                       dx=dx)
            dae_list.extend([Ode('SDM of ' + self.name + ' 1',
                                 eqn_dict['Ode'][0][0],
                                 eqn_dict['Ode'][0][1]),
                             Ode('SDM of ' + self.name + ' 2',
                                 eqn_dict['Ode'][1][0],
                                 eqn_dict['Ode'][1][1]),
                             Ode('SDM of ' + self.name + ' 3',
                                 eqn_dict['Ode'][2][0],
                                 eqn_dict['Ode'][2][1]),
                             Eqn('minmod limiter 1 of ' + u.name,
                                 eqn_dict['Eqn'][0]),
                             Eqn('minmod limiter 2 of ' + u.name,
                                 eqn_dict['Eqn'][1]),
                             Eqn('minmod limiter 3 of ' + u.name,
                                 eqn_dict['Eqn'][2])])
        elif scheme == 'TVD1':
            eqn_dict = semi_descritize(self.diff_var,
                                       self.flux,
                                       self.source,
                                       self.two_dim_var,
                                       M,
                                       scheme='TVD1',
                                       a0=a0,
                                       a1=a1,
                                       dx=dx)
            dae_list.extend([Ode('SDM of ' + self.name + 'using',
                                 eqn_dict['Ode'][0],
                                 eqn_dict['Ode'][1])])
        else:
            raise NotImplementedError(f'Scheme {scheme} not implemented')

        if output_boundary:
            dae_list.extend([Eqn(f'Equation of {u[M]}', u[M] - 2 * iVar(u.name + 'R') + u[M - 1]),
                             Eqn(f'Equation of {u[0]}', u[0] - 2 * iVar(u.name + 'L') + u[1])])

        return dae_list
