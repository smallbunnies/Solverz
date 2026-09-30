"""The Integrator: the state of one integration and the services its
algorithm uses to take a step."""
import inspect
import weakref

import numpy as np

from Solverz.solvers.stats import Stats
from Solverz.integrator.algorithm import StepContext, StepFailure, check_style
from Solverz.integrator.derivative import DFDT_POLICIES
from Solverz.integrator.linalg import IterationMatrix
from Solverz.integrator.options import IntegratorOptions
from Solverz.integrator.policies import DefaultPolicy, LegacyRodasPolicy

__all__ = ['Integrator']

# Whether a residual takes ``out``, inspected once per residual object, since
# EventLoop starts one call per segment on the same model.
_TAKES_OUT = weakref.WeakKeyDictionary()


def _takes_out(F):
    """Whether ``F`` has a parameter named ``out``.

    The signature is read with ``follow_wrapped=False``, so that the adapter
    ``nDAE`` puts around a residual without ``out`` is recognized by its own
    signature instead of that of the residual it wraps.
    """
    try:
        return _TAKES_OUT[F]
    except (KeyError, TypeError):
        pass
    try:
        takes = 'out' in inspect.signature(getattr(F, 'py_func', F), follow_wrapped=False).parameters
    except (TypeError, ValueError):
        takes = False
    try:
        _TAKES_OUT[F] = takes
    except TypeError:
        pass
    return takes


def _residual(F):
    """``F`` when it takes ``out``; otherwise an adapter that copies its result into ``out``.

    ``_with_out`` of ``num_eqn`` is not used here: it reads the signature
    through ``functools.wraps``, and would return unchanged a wrapper without
    ``out`` around a residual that has it.
    """
    if _takes_out(F):
        return F

    def F_(t, y, p, out=None):
        r = F(t, y, p)
        if out is None:
            return r
        out[...] = r
        return out

    return F_


class Integrator:
    """One integration: the state of the run and the services of a step.

    ``u`` and ``uprev`` are float64 vectors owned by the Integrator and never
    rebound. An attempt reads ``t``, ``dt`` and ``uprev`` and writes ``u``
    and, in an adaptive run, ``EEst``. After acceptance the step runs from
    ``(tprev, uprev)`` to ``(t_step, u_step)`` and has the length ``dt_step``;
    ``interp`` and the algorithm's interpolant describe it through these
    fields, never through ``u``, which an event may replace.

    The services ``F``, ``J``, ``F0``, ``J0``, ``dFdt`` and ``W`` count
    every evaluation and factorization they make in ``stats``. ``F0``, ``J0``
    and ``dFdt`` belong to the start of the step: each is evaluated on its
    first request after a new step began and kept on the retries of that
    step. ``W(gamma)`` is factorized once per ``gamma`` per attempt.
    """

    def __init__(self, dae, tspan, y0, alg, opt=None):
        check_style(alg)
        self.dae = dae
        self.alg = alg
        self.opts = opts = IntegratorOptions.from_opt(opt, alg, tspan)
        self.policy = (LegacyRodasPolicy if opts.legacy_compat else DefaultPolicy)(opts, alg)
        self._dfdt = DFDT_POLICIES[self.policy.dfdt]
        self._error_norm = self.policy.error_norm

        # a copy: the caller's array is never written
        self.u = np.array(y0, dtype=np.float64)
        self.n = n = self.u.shape[0]
        self.M = dae.M
        self.p = dae.p
        self.model_epoch = 0
        self._F = _residual(dae.F)
        self._J = dae.J
        self.stats = Stats(alg.scheme)
        self.stats.ncondition = 0
        self.linalg = IterationMatrix(self)

        self.t0, self.tend = opts.t0, opts.tend
        self.t = self.tprev = self.t_step = opts.t0
        self.dt = self.dt_step = None
        self.uprev = self.u.copy()
        self.u_step = self.u
        self.EEst = None
        self.new_step = True
        self.force_stepfail = False
        self._stepfail_reason = None

        # the start of the current step, at which F0, J0 and dF/dt are taken:
        # t during an attempt, tprev after acceptance
        self._t_start = opts.t0
        self._J0 = None
        self._F0 = np.empty(n)
        self._F0_ro = self._F0.view()
        self._F0_ro.flags.writeable = False
        self._F0_valid = False
        self._ft = np.empty(n)
        self._ft_ro = self._ft.view()
        self._ft_ro.flags.writeable = False
        self._ft_valid = False
        self._ft_scratch = np.empty(n)
        self._W_cache = {}
        self._w = np.empty(n)
        self._w2 = np.empty(n)
        self._fin = np.empty(n, dtype=bool)
        self._interp_ready = False
        self._interp_valid = True

        self.ctx = StepContext(self)
        self.cache = alg.alloc(self)

    # -- services -----------------------------------------------------------

    def F(self, t, y, out=None):
        """The residual ``F(t, y)``, into ``out`` or into a new array."""
        if out is None:
            out = np.empty(self.n)
        self.stats.nfeval += 1
        try:
            self._F(t, y, self.p, out=out)
        except ArithmeticError as e:
            raise StepFailure(f"{type(e).__name__} in F: {e}") from e
        return out

    def J(self, t, y):
        """The Jacobian of ``F`` at ``(t, y)``, evaluated at every call."""
        self.stats.nJeval += 1
        try:
            return self._J(t, y, self.p)
        except ArithmeticError as e:
            raise StepFailure(f"{type(e).__name__} in J: {e}") from e

    def F0(self):
        """A read-only view of ``F`` at the start of the step."""
        if not self._F0_valid:
            self.F(self._t_start, self.uprev, out=self._F0)
            self._F0_valid = True
        return self._F0_ro

    def J0(self):
        """The Jacobian at the start of the step, the matrix of ``W``."""
        J = self._J0
        if J is None:
            J = self._J0 = self.J(self._t_start, self.uprev)
        return J

    def dFdt(self, out=None):
        """``dF/dt`` at the start of the step by the configuration's policy;
        a read-only view, or a copy in ``out``.

        The increment of ``'ode23s'`` scales with the step of the first
        attempt, the one that computes it.
        """
        if not self._ft_valid:
            self._dfdt(self.F, self._t_start, self.dt, self.uprev, self.F0(), self._ft, self._ft_scratch)
            self._ft_valid = True
        if out is None:
            return self._ft_ro
        np.copyto(out, self._ft)
        return out

    def W(self, gamma):
        """The factorization of ``M - (dt*gamma) J0`` for the current attempt."""
        W = self._W_cache.get(gamma)
        if W is None:
            W = self._W_cache[gamma] = self.linalg.factorize(self.M, self.J0(), self.dt, gamma)
        return W

    def error_norm(self, e):
        """The scalar error of the vector ``e`` in the configuration's norm."""
        return self._error_norm(self, e)

    # -- one attempt --------------------------------------------------------

    def perform_step(self):
        """One attempt of the algorithm, in its style.

        A ``StepFailure``, or an ``ArithmeticError`` such as NumPy raises
        under ``np.seterr(all='raise')``, fails the attempt: ``force_stepfail``
        is set and ``_stepfail_reason`` holds the message. A contract error of
        the algorithm raises ``TypeError``.
        """
        alg = self.alg
        self._W_cache.clear()
        if self.new_step:
            self._t_start = self.t
            self._J0 = None
            self._F0_valid = False
            self._ft_valid = False
        self._interp_ready = False
        self.EEst = None
        try:
            if alg.inplace:
                alg.perform_step(self, self.cache)
            else:
                self._formula_step()
        except StepFailure as e:
            self.force_stepfail = True
            self._stepfail_reason = str(e)
        except ArithmeticError as e:
            self.force_stepfail = True
            self._stepfail_reason = f"{type(e).__name__}: {e}"
        else:
            if alg.inplace and self.opts.adaptive and self.EEst is None:
                raise TypeError(f"{alg.scheme}.perform_step set no error estimate")
        self.new_step = False

    def _formula_step(self):
        alg = self.alg
        res = alg.perform_step(self.ctx)
        y, err = res if isinstance(res, tuple) else (res, None)
        n = self.n
        if np.shape(y) != (n,):
            raise TypeError(f"{alg.scheme}.perform_step returned y of shape {np.shape(y)}; "
                            f"it must be a vector of length {n}")
        if err is not None and np.shape(err) != (n,):
            raise TypeError(f"{alg.scheme}.perform_step returned an error estimate of shape "
                            f"{np.shape(err)}; it must be a vector of length {n}")
        np.copyto(self.u, y)
        if self.opts.adaptive:
            if err is None:
                raise TypeError(f"{alg.scheme}.perform_step returned no error estimate; "
                                f"set adaptive = False or run with opt.fix_h")
            self.EEst = self.error_norm(err)
        elif err is not None and not alg.adaptive:
            raise TypeError(f"{alg.scheme}.perform_step returned an error estimate but declares "
                            f"adaptive = False")

    # -- dense output -------------------------------------------------------

    def interp(self, tq, out=None):
        """The state at ``tq`` inside the last accepted step, into ``out`` or a new array.

        ``tq`` must lie in ``[tprev, t]``; the two ends return copies of
        ``uprev`` and ``u`` exactly, and after the state or the model changed
        outside a step only ``tq == t`` is accepted. Otherwise the algorithm's
        ``addsteps`` runs once per step and its interpolant is evaluated at
        ``theta = (tq - tprev) / dt_step``, which may exceed 1 by one rounding
        after a step that landed on a stop time.
        """
        if out is None:
            out = np.empty(self.n)
        if tq == self.t:
            np.copyto(out, self.u)
            return out
        if not self._interp_valid:
            raise ValueError(f"the state was changed at t = {self.t!r} after the last step; "
                             f"only tq == t can be interpolated until the next step")
        if not self.tprev <= tq <= self.t:
            raise ValueError(f"tq = {tq!r} lies outside the last step [{self.tprev!r}, {self.t!r}]")
        if tq == self.tprev:
            np.copyto(out, self.uprev)
            return out
        alg = self.alg
        if not self._interp_ready:
            alg.addsteps(self, self.cache)
            self._interp_ready = True
        r = alg.interpolant(self, self.cache, (tq - self.tprev) / self.dt_step, out)
        if r is not None and r is not out:
            np.copyto(out, r)
        return out
