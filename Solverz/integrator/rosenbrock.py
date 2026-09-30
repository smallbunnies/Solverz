"""The Rosenbrock family: a method is its tableau, and one in-place
``perform_step`` takes the step of every table."""
from types import SimpleNamespace

import numpy as np
from scipy.sparse import issparse

from Solverz.solvers.daesolver.rodas.param import Rodas_param
from Solverz.solvers.daesolver.rodas.rodas import ntrp1, ntrp2
from Solverz.integrator.algorithm import Algorithm

__all__ = ['RosenbrockTableau', 'Rosenbrock', 'Rodas3', 'Rodas4', 'Rodasp', 'Rodas5P']

_INTERPOLATIONS = ('ntrp1', 'hermite', 'linear')


def _has_dense_output(tab):
    return all(getattr(tab, k, None) is not None for k in ('c', 'd', 'e'))


class RosenbrockTableau:
    """The coefficients of a Rosenbrock method, with the attributes of ``Rodas_param``.

    ``s`` stages, ``pord`` the order of the method; ``alpha`` and
    ``gammatilde`` are stored transposed, so ``alpha[:, j]`` is row ``j`` of
    the table; ``a`` holds the stage times and ``g`` the coefficients of
    ``dF/dt``; ``b`` and ``bd`` are the weights of the solution and of the
    embedded solution; ``c``, ``d`` and ``e`` are the coefficients of the
    dense output, or ``None``.
    """

    def __init__(self, *, s, pord, gamma, alpha, gammatilde, a, g, b, bd, c=None, d=None, e=None):
        self.s = s
        self.pord = pord
        self.gamma = gamma
        self.alpha = alpha
        self.gammatilde = gammatilde
        self.a = a
        self.g = g
        self.b = b
        self.bd = bd
        self.c = c
        self.d = d
        self.e = e

    @classmethod
    def from_hairer(cls, gamma, alpha, *, beta=None, gamma_ij=None, b, bd, pord, c=None, d=None, e=None):
        """The tableau of a method published in Hairer's notation.

        ``alpha`` is the strictly lower triangular ``s x s`` table
        ``alpha_ij``. Exactly one of ``beta``, the strictly lower triangular
        table of ``alpha_ij + gamma_ij``, and ``gamma_ij``, the coupling
        coefficients as published for ROS methods, is given. ``c``, ``d`` and
        ``e`` are given together or not at all. The derived coefficients are
        computed with the expressions of ``Rodas_param``, so the ``rodas4``
        tables give its arrays bit for bit.
        """
        if (beta is None) == (gamma_ij is None):
            raise TypeError("from_hairer takes exactly one of beta and gamma_ij")
        alpha = np.array(alpha, dtype=np.float64)
        s = alpha.shape[0]
        coupling = np.array(beta if beta is not None else gamma_ij, dtype=np.float64)
        for name, table in (('alpha', alpha), ('beta' if beta is not None else 'gamma_ij', coupling)):
            if table.shape != (s, s):
                raise ValueError(f"{name} must be a square table of shape ({s}, {s}), not {table.shape}")
            if np.any(np.triu(table) != 0):
                raise ValueError(f"{name} must be strictly lower triangular")
        b = np.array(b, dtype=np.float64)
        bd = np.array(bd, dtype=np.float64)
        dense = [x is not None for x in (c, d, e)]
        if any(dense) and not all(dense):
            raise ValueError("c, d and e are given together or not at all")
        if all(dense):
            c, d, e = (np.array(x, dtype=np.float64) for x in (c, d, e))
        for name, v in (('b', b), ('bd', bd), ('c', c), ('d', d), ('e', e)):
            if v is not None and v.shape != (s,):
                raise ValueError(f"{name} must be a vector of length {s}, not of shape {v.shape}")
        gamma = float(gamma)
        gammatilde = coupling - alpha if beta is not None else coupling
        a = np.sum(alpha, axis=1)
        g = np.sum(gammatilde, axis=1) + gamma
        gammatilde = gammatilde / gamma
        return cls(s=s, pord=pord, gamma=gamma, alpha=alpha.T, gammatilde=gammatilde.T, a=a, g=g,
                   b=b, bd=bd, c=c, d=d, e=e)


def _pairing(M):
    """``(rows, cols, Mv)``: the nonzero entries of ``M``, explicit zeros
    dropped, when no row and no column holds two of them; otherwise ``None``.

    Variable ``cols[k]`` is then differential with ``y' = F[rows[k]] / Mv[k]``.
    """
    if issparse(M):
        coo = M.tocoo()
        keep = coo.data != 0
        rows, cols, Mv = coo.row[keep], coo.col[keep], coo.data[keep]
    else:
        M = np.asarray(M)
        rows, cols = np.nonzero(M)
        Mv = M[rows, cols]
    if np.unique(rows).size != rows.size or np.unique(cols).size != cols.size:
        return None
    return rows, cols, np.asarray(Mv, dtype=np.float64)


class Rosenbrock(Algorithm):
    """A Rosenbrock method, defined by its ``tableau``.

    A subclass states ``scheme`` and ``tableau``, a ``RosenbrockTableau`` or
    any object with the attributes of ``Rodas_param``; ``order`` and
    ``error_order`` are then the tableau's ``pord``. ``interpolation`` is
    ``'ntrp1'``, the dense output of the tableau's ``c``, ``d`` and ``e``,
    ``'hermite'``, the cubic Hermite interpolant between the two ends of the
    step and their slopes, or ``'linear'``; a subclass that does not state it
    gets ``'ntrp1'`` when the tableau has ``c``, ``d`` and ``e`` and
    ``'linear'`` otherwise.

    ``legacy_compat=True`` selects the configuration that reproduces legacy
    Rodas. The argument is keyword-only, so that a class passed where an
    instance is expected fails instead of selecting a configuration.
    """

    inplace = True
    adaptive = True
    norm = 'max'
    tableau = None
    interpolation = 'linear'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        tab = cls.__dict__.get('tableau')
        if tab is not None:
            if 'order' not in cls.__dict__:
                cls.order = tab.pord
            if 'error_order' not in cls.__dict__:
                cls.error_order = tab.pord
            if 'interpolation' not in cls.__dict__:
                cls.interpolation = 'ntrp1' if _has_dense_output(tab) else 'linear'
        if cls.interpolation not in _INTERPOLATIONS:
            raise TypeError(f"{cls.__name__}.interpolation is {cls.interpolation!r}; "
                            f"it must be one of {_INTERPOLATIONS}")
        if cls.interpolation == 'ntrp1' and cls.tableau is not None and not _has_dense_output(cls.tableau):
            raise TypeError(f"{cls.__name__}.interpolation is 'ntrp1', but its tableau has no c, d and e")

    def __init__(self, *, legacy_compat=False):
        if self.tableau is None:
            raise TypeError(f"{type(self).__name__} has no tableau; use a method such as Rodas4(), "
                            f"or subclass Rosenbrock with a tableau")
        self.legacy_compat = bool(legacy_compat)

    @classmethod
    def from_scheme(cls, name, *, legacy_compat=False):
        """The method that ``opt.scheme`` names in legacy Rodas."""
        if name == 'rodas3d':
            raise ValueError("'rodas3d' defines no dense-output coefficients (param.py:171-200) "
                             "and is not provided by the integrator core")
        try:
            method = _BY_SCHEME[name]
        except (KeyError, TypeError):
            raise ValueError(f"unknown Rosenbrock scheme {name!r}") from None
        return method(legacy_compat=legacy_compat)

    def alloc(self, integ):
        """The stage matrix ``K`` and the work vectors of a step, allocated once.

        ``K`` is C-contiguous of shape ``(n, s)``, the layout of legacy Rodas,
        so that every stage product is the same BLAS call and the compiled
        dense output receives the same specialization.
        """
        tab = self.tableau
        n, s = integ.n, tab.s
        c = SimpleNamespace(K=np.zeros((n, s)),
                            dtb=np.empty(s), dtbd=np.empty(s),
                            alpha_cols=[tab.alpha[:, j] for j in range(s)],
                            gt_cols=[tab.gammatilde[:, j] for j in range(s)])
        for name in ('Fs', 'dfdt0', 'rhs', 'tmp', 'sum1', 'sum2', 'y1', 'utilde', 'x'):
            setattr(c, name, np.empty(n))
        if self.interpolation == 'hermite':
            c.F1, c.s0, c.s1 = np.empty(n), np.empty(n), np.empty(n)
            c.pairing, c.pairing_epoch = None, None
        return c

    def perform_step(self, integ, cache):
        """One attempt: ``integ.u`` and, in an adaptive run, ``integ.EEst``.

        Every operation is the one of legacy Rodas with the same operands in
        the same association, so the result is legacy's bit for bit;
        multiplication and addition of two operands commute in IEEE
        arithmetic. ``K`` is zeroed at every attempt, since a stale non-finite
        column would turn ``0 * inf`` into ``NaN`` in the stage products, and
        the error is the difference of two products, never ``K @ (dt*(b - bd))``.
        """
        tab = self.tableau
        c = cache
        t, dt, y0, M = integ.t, integ.dt, integ.uprev, integ.M
        K, rhs, tmp, x = c.K, c.rhs, c.tmp, c.x
        sum1, sum2, y1, Fs, dfdt0 = c.sum1, c.sum2, c.y1, c.Fs, c.dfdt0
        g, a = tab.g, tab.a

        F0 = integ.F0()
        ft = integ.dFdt()
        K.fill(0.0)
        np.multiply(ft, dt, out=dfdt0)
        np.multiply(dfdt0, g[0], out=tmp)
        np.add(F0, tmp, out=rhs)
        W = integ.W(tab.gamma)
        W.solve(rhs, out=x)
        K[:, 0] = x
        for j in range(1, tab.s):
            np.matmul(K, c.alpha_cols[j], out=sum1)
            np.matmul(K, c.gt_cols[j], out=sum2)
            np.multiply(sum1, dt, out=y1)
            np.add(y0, y1, out=y1)
            integ.F(t + dt * a[j], y1, out=Fs)
            # the product allocates, as in legacy: for a sparse M SciPy
            # accumulates it into fresh zeros, which turns -0.0 into +0.0, and
            # a product into a buffer need not give the same bits
            np.add(Fs, M @ sum2, out=rhs)
            np.multiply(dfdt0, g[j], out=tmp)
            np.add(rhs, tmp, out=rhs)
            W.solve(rhs, out=x)
            np.subtract(x, sum2, out=K[:, j])
        np.multiply(tab.b, dt, out=c.dtb)
        np.matmul(K, c.dtb, out=sum1)
        np.add(y0, sum1, out=integ.u)
        if integ.opts.adaptive:
            np.multiply(tab.bd, dt, out=c.dtbd)
            np.matmul(K, c.dtbd, out=sum2)
            np.subtract(sum1, sum2, out=c.utilde)
            integ.EEst = integ.error_norm(c.utilde)

    def addsteps(self, integ, cache):
        """The slopes of the Hermite interpolant at the two ends of the step.

        The end residual ``F(t_step, u_step)`` costs one residual per
        interpolated step. In the legacy-compatible configuration the slopes
        are the raw residuals at the two ends, as legacy Rodas uses them. In
        the default configuration a residual row is the slope of the variable
        it is paired with through ``M``, divided by the entry of ``M``; every
        other variable, and every variable when some row or column of ``M``
        holds two nonzeros, gets the secant slope, which makes the
        interpolant linear in it. Row ``i`` of ``F`` is not the derivative of
        variable ``i`` unless ``M`` is the identity, and an algebraic variable
        has no derivative in ``F`` at all.
        """
        if self.interpolation != 'hermite':
            return
        c = cache
        if integ.opts.legacy_compat:
            integ.F(integ.t_step, integ.u_step, out=c.s1)
            np.copyto(c.s0, integ.F0())
            return
        integ.F(integ.t_step, integ.u_step, out=c.F1)
        F0 = integ.F0()
        if c.pairing_epoch != integ.model_epoch:
            c.pairing = _pairing(integ.M)
            c.pairing_epoch = integ.model_epoch
        np.subtract(integ.u_step, integ.uprev, out=c.s0)
        np.divide(c.s0, integ.dt_step, out=c.s0)
        np.copyto(c.s1, c.s0)
        if c.pairing is not None:
            rows, cols, Mv = c.pairing
            c.s0[cols] = F0[rows] / Mv
            c.s1[cols] = c.F1[rows] / Mv

    def interpolant(self, integ, cache, theta, out):
        """The state at ``tprev + theta * dt_step``, into ``out``.

        ``'ntrp1'`` and ``'hermite'`` call the compiled functions of legacy
        Rodas: their products run through Numba's BLAS binding, which a NumPy
        expression of the same association need not reproduce bit for bit.
        """
        if self.interpolation == 'ntrp1':
            tab = self.tableau
            out[...] = ntrp1(integ.uprev, theta, integ.dt_step, cache.K, tab.b, tab.c, tab.d, tab.e)
        elif self.interpolation == 'hermite':
            out[...] = ntrp2(integ.uprev, integ.u_step, cache.s0, cache.s1, theta, integ.dt_step)
        else:
            super().interpolant(integ, cache, theta, out)


class Rodas3(Rosenbrock):
    """Rodas3, of order 3, with the cubic Hermite interpolant."""

    scheme = 'rodas3'
    tableau = Rodas_param('rodas3')
    interpolation = 'hermite'
    interp_order = 2


class Rodas4(Rosenbrock):
    """Rodas4 of Hairer and Wanner, of order 4, with its dense output of order 3.

    E. Hairer and G. Wanner, Solving Ordinary Differential Equations II, 2nd
    ed., Springer, 1996.
    """

    scheme = 'rodas4'
    tableau = Rodas_param('rodas4')
    interpolation = 'ntrp1'
    interp_order = 3


class Rodasp(Rosenbrock):
    """Rodasp of Steinebach, of order 4, with its dense output of order 3.

    G. Steinebach, Order-reduction of ROW-methods for DAEs and method of
    lines applications, Preprint 1741, FB Mathematik, TH Darmstadt, 1995.
    """

    scheme = 'rodasp'
    tableau = Rodas_param('rodasp')
    interpolation = 'ntrp1'
    interp_order = 3


class Rodas5P(Rosenbrock):
    """Rodas5P of Steinebach, of order 5, with its dense output.

    G. Steinebach, Construction of Rosenbrock-Wanner method Rodas5P and
    numerical benchmarks within the Julia Differential Equations package,
    BIT 63, 27, 2023.
    """

    scheme = 'rodas5p'
    tableau = Rodas_param('rodas5p')
    interpolation = 'ntrp1'
    interp_order = 3


_BY_SCHEME = {m.scheme: m for m in (Rodas3, Rodas4, Rodasp, Rodas5P)}
