"""The Integrator: the state of one integration, the loop that advances it,
and the services its algorithm uses to take a step."""
import inspect
import math
import weakref
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.sparse import issparse

from Solverz.solvers.daesolver.daeic import DaeIc
from Solverz.solvers.laesolver import get_linsolver, linsolver
from Solverz.solvers.stats import Stats
from Solverz.variable.variables import Vars
from Solverz.integrator.algorithm import StepContext, StepFailure, check_style, _warn_scheme
from Solverz.integrator.callbacks import (ContinuousCallback, DiscreteCallback, _ContinuousState,
                                          _LegacyEventCallback, locate)
from Solverz.integrator.derivative import DFDT_POLICIES
from Solverz.integrator.linalg import IterationMatrix
from Solverz.integrator.nlsolve import implicit as _implicit
from Solverz.integrator.options import IntegratorOptions
from Solverz.integrator.policies import DefaultPolicy, LegacyRodasPolicy
from Solverz.integrator.rosenbrock import Rodas4, _pairing
from Solverz.integrator.saving import EventLog, SolutionBuffer, to_daesol

__all__ = ['Integrator', 'init', 'solve']

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


def _one_to_one(M, n):
    """``(rows, Mv)`` with ``y'[c] = F[rows[c]] / Mv[c]`` for every variable
    ``c``, when ``M`` pairs every row with exactly one variable and every
    variable with exactly one row; otherwise ``None``.

    Such an ``M`` is a scaled permutation, the mass matrix of an ODE whose
    equations may be declared in any order. ``rows`` is a contiguous
    ``intp`` array, so that ``np.take`` gathers without a copy of it.
    """
    pairing = _pairing(M)
    if pairing is None or pairing[0].size != n:
        return None
    rows, cols, Mv = pairing
    order = np.argsort(cols)
    return np.ascontiguousarray(rows[order], dtype=np.intp), np.ascontiguousarray(Mv[order])


def _differential_rows(M, n):
    """1.0 on the rows of ``M`` that hold a nonzero value, 0.0 on the others,
    explicit zeros counting as zero; read-only."""
    D = np.zeros(n)
    if issparse(M):
        coo = M.tocoo()
        D[coo.row[coo.data != 0]] = 1.0
    else:
        D[np.any(np.asarray(M) != 0, axis=1)] = 1.0
    D.flags.writeable = False
    return D


def solve(dae, tspan, y0, alg=None, opt=None, *, callbacks=(), tstops=()):
    """Integrate ``dae`` over ``tspan`` from ``y0`` and return a ``daesol``.

    ``alg`` is the method, ``Rodas4()`` when ``None``; ``opt`` holds the
    tolerances and step bounds, ``Opt()`` when ``None``, and is never
    written. ``tspan`` has the legacy meaning: ``[t0, tend]`` saves every
    accepted step, and more entries save at ``tspan[1:]`` by interpolation.
    ``tstops`` are times at which a step must end exactly. A ``Vars``
    ``y0`` gives ``TimeVars`` rows. A run that fails returns the rows saved
    so far with ``stats.ret == 'failed'`` and prints one line; it never
    raises.
    """
    if alg is None:
        alg = Rodas4()
    _warn_scheme(opt, alg)
    if opt is not None and opt.profile:
        start = perf_counter()
        sol = Integrator(dae, tspan, y0, alg, opt, callbacks, tstops).solve()
        end = perf_counter()
        print(f"Time elapsed: {end - start}s")
        return sol
    return Integrator(dae, tspan, y0, alg, opt, callbacks, tstops).solve()


def init(dae, tspan, y0, alg=None, opt=None, *, callbacks=(), tstops=()):
    """The ``Integrator`` of ``solve(dae, tspan, y0, alg, opt, ...)``, before
    its first step; ``integ.step()`` advances it by one accepted step and
    ``integ.solve()`` runs it to the end."""
    if alg is None:
        alg = Rodas4()
    _warn_scheme(opt, alg)
    return Integrator(dae, tspan, y0, alg, opt, callbacks, tstops)


class Integrator:
    """One integration: the state of the run, its loop and the services of a step.

    ``u`` and ``uprev`` are float64 vectors owned by the Integrator and never
    rebound. An attempt reads ``t``, ``dt`` and ``uprev`` and writes ``u``
    and, in an adaptive run, ``EEst``. After acceptance the step runs from
    ``(tprev, uprev)`` to ``(t_step, u_step)`` and has the length ``dt_step``;
    ``interp`` and the algorithm's interpolant describe it through these
    fields, never through ``u``, which an event may replace. The accepted
    state is committed into ``uprev`` at the start of the next attempt, so
    between two ``step()`` calls both ends of the step are available.

    The services ``F``, ``f``, ``J``, ``F0``, ``J0``, ``dFdt``, ``W`` and
    ``implicit`` count every evaluation, factorization and solve they make
    in ``stats``. ``F0``, ``J0`` and ``dFdt`` belong to the start of the
    step: each is evaluated on its first request after a new step began and
    kept on the retries of that step. ``W(gamma)`` is factorized once per
    ``gamma`` per attempt. ``ctx`` offers them to a formula algorithm.

    The loop follows OrdinaryDiffEq.jl: ``loopheader`` commits the accepted
    step or applies the controller's rejection and checks the proposed step,
    ``perform_step`` is the algorithm's attempt, and ``loopfooter`` accepts
    or rejects it, advances ``t``, handles the callbacks and saves. The
    configuration's policy decides the bounds, the saving and the end test,
    so the loop has no branch on the configuration.

    The state or the model changes inside a call in two places only: in a
    callback's affect, and between two ``step()`` calls followed by
    ``model_modified()``. Both run the modification protocol, which makes
    the state consistent with ``DaeIc`` and discards everything derived from
    the old state or model.
    """

    def __init__(self, dae, tspan, y0, alg, opt=None, callbacks=(), tstops=()):
        check_style(alg)
        self.dae = dae
        self.alg = alg
        self.opts = opts = IntegratorOptions.from_opt(opt, alg, tspan)
        callbacks, tstops = tuple(callbacks), tuple(tstops)
        if opts.legacy_compat and (tstops or any(len(getattr(cb, 'tstops', ())) for cb in callbacks)):
            raise ValueError(f"{type(alg).__name__}(legacy_compat=True) reproduces legacy Rodas, "
                             f"which has no tstops; pass tstops to {type(alg).__name__}()")
        continuous, discrete = [], []
        for cb in callbacks:
            if isinstance(cb, ContinuousCallback):
                continuous.append(cb)
            elif isinstance(cb, DiscreteCallback):
                discrete.append(cb)
            else:
                raise TypeError(f"callbacks holds ContinuousCallback and DiscreteCallback objects, "
                                f"not {type(cb).__name__}")
        if opts.event is not None:
            continuous.append(_LegacyEventCallback(opts.event))
        if sum(cb.record for cb in continuous) > 1:
            raise ValueError("at most one continuous callback of a run may record its crossings, "
                             "and opt.event records its events")
        self.policy = policy = (LegacyRodasPolicy if opts.legacy_compat else DefaultPolicy)(opts, alg)
        self._dfdt = DFDT_POLICIES[policy.dfdt]
        self._error_norm = policy.error_norm
        self.t0, self.tend = opts.t0, opts.tend

        if isinstance(y0, Vars):
            self._address = y0.a
            y0 = y0.array
        else:
            self._address = None
        # a copy: the caller's array is never written, and DaeIc returns it
        # unchanged when the initial state is consistent
        self.u = np.array(y0, dtype=np.float64)
        self.n = n = self.u.shape[0]
        self.M = dae.M
        self.p = dae.p
        self.model_epoch = 0
        self._F = _residual(dae.F)
        self._J = dae.J
        # caches derived from M, each valid for the model_epoch it was built for
        self._D = self._D_epoch = None
        self._f_pairing = self._f_epoch = None
        if alg.explicit:
            self._f_pairing, self._f_epoch = _one_to_one(self.M, n), self.model_epoch
            if self._f_pairing is None:
                raise TypeError(f"{alg.scheme} is explicit and cannot integrate a model with algebraic "
                                f"equations or a singular mass matrix")
        self.stats = Stats(alg.scheme)
        self.stats.ncondition = 0
        self.linalg = IterationMatrix(self)
        self._daeic_backend = get_linsolver()

        self.t = self.tprev = self.t_step = opts.t0
        self.dt = self.dt_step = None
        self.EEst = None
        self.new_step = True
        self.force_stepfail = False
        self._stepfail_reason = None
        self._skip_step = False
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
        self._nl_eta = 1.0
        self._nl_y, self._nl_G, self._nl_dz = np.empty(n), np.empty(n), np.empty(n)
        self._nl_w, self._nl_w2 = np.empty(n), np.empty(n)
        self._f_scratch = np.empty(n)
        self._u_step_buf = np.empty(n)
        self._cb_y = np.empty(n)
        self._te = None

        y, reason = self._daeic(self.u, opts.t0)
        if y is not None and y is not self.u:
            np.copyto(self.u, y)
        self.uprev = self.u.copy()
        self.u_step = self.u
        self.sol = SolutionBuffer()
        self.events = EventLog()
        self.sol.push(opts.t0, self.u.copy())
        self.saveat = opts.saveat
        self.saveat_idx = 0

        self.ctx = StepContext(self)
        self.controller = alg.controller(opts)
        self.cache = alg.alloc(self)

        if opts.legacy_compat:
            self.tstops = []
        else:
            t0, tend = opts.t0, opts.tend
            stops = list(tstops)
            for cb in discrete:
                stops += cb.tstops
            # sorted, and so already a heap, with tend its largest entry
            self.tstops = sorted({float(x) for x in stops if t0 < x < tend})
            if tend > t0:
                self.tstops.append(tend)
        self.next_step_tstop = False
        self.tstop_target = None
        self.dt_untruncated = None

        # after DaeIc, so that the bottom values are those of the consistent initial state
        self.continuous_callbacks = [_ContinuousState(cb, k, self) for k, cb in enumerate(continuous)]
        self.discrete_callbacks = discrete

        self._pbar = None
        if opts.pbar:
            # created after DaeIc, so that an error DaeIc propagates leaves no open bar
            from tqdm import tqdm
            self._pbar = tqdm(total=self.tend - self.t0)
        self.dt = policy.initial_dt(self, alg.initial_dt(self))
        self.dtpropose = self.dt
        self.q = None
        self.iter = 0
        self.accept_step = False
        self.nconsecutive_reject = 0
        self.terminated = self.failed = self.finished = False
        self.retcode = None
        if reason is not None:
            self._fail(reason)
        else:
            self.finished = bool(policy.at_end(self))

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

    def f(self, t, y, out=None):
        """``M^-1 F(t, y)``, into ``out`` or into a new array; one counted residual.

        ``M`` must pair every row with exactly one variable, which an
        explicit algorithm is guaranteed; otherwise ``TypeError``. The pairing
        is found once per ``model_epoch``, and the derivative of variable
        ``c`` is the residual of its row divided by its entry of ``M``.
        """
        if self._f_epoch != self.model_epoch:
            self._f_pairing, self._f_epoch = _one_to_one(self.M, self.n), self.model_epoch
        pairing = self._f_pairing
        if pairing is None:
            raise TypeError(f"{self.alg.scheme}: f = M^-1 F needs a mass matrix that pairs every row "
                            f"with exactly one variable; this model has algebraic equations or a "
                            f"singular mass matrix")
        rows, Mv = pairing
        Fy = self.F(t, y, out=self._f_scratch)
        if out is None:
            out = np.empty(self.n)
        # mode='clip' gathers straight into out; 'raise' would buffer it
        np.take(Fy, rows, out=out, mode='clip')
        np.divide(out, Mv, out=out)
        return out

    def D(self):
        """1.0 on the differential rows of ``M``, 0.0 on the algebraic rows,
        whose values are all zero, explicit zeros included; read-only, built
        once per ``model_epoch``."""
        if self._D_epoch != self.model_epoch:
            self._D, self._D_epoch = _differential_rows(self.M, self.n), self.model_epoch
        return self._D

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

    def implicit(self, t, gamma, rhs, y=None, out=None, slope=False):
        """``y`` with ``M y - (dt*gamma) F(t, y) = rhs`` by the simplified Newton
        iteration of ``nlsolve.py``, from ``y`` or ``uprev``, with the Newton
        matrix ``W(gamma)``; ``(y, k)`` with ``slope=True``, where ``k = (M y -
        rhs) / (dt*gamma)`` equals ``F(t, y)`` to the Newton tolerance. A
        failed iteration raises ``StepFailure``."""
        return _implicit(self, t, gamma, rhs, y, out, slope)

    def error_norm(self, e):
        """The scalar error of the vector ``e`` in the configuration's norm."""
        return self._error_norm(self, e)

    # -- initial values -----------------------------------------------------

    def _daeic_F(self, t, y, p):
        return self.F(t, y)

    def _daeic_J(self, t, y, p):
        return self.J(t, y)

    def _daeic(self, y, t):
        """``(y, None)`` with ``y`` consistent at ``t`` by ``DaeIc``, or ``(None,
        reason)`` when ``DaeIc`` fails.

        ``DaeIc`` runs on a proxy of the model whose residual and Jacobian
        are the counted services, and with the backend that was global at
        initialization, so that a later call outside the caller's ``with
        linsolver(...)`` block uses the same one. Four failures end the run:
        ``'Need Better y0'``, a ``LinAlgError`` or a ``RuntimeError`` from a
        singular algebraic Jacobian, and a ``StepFailure`` from an arithmetic
        error of ``F`` or ``J``. Any other exception is a programming error
        and propagates; a ``RuntimeError`` raised by the model itself cannot
        be told apart from a solver failure.
        """
        proxy = SimpleNamespace(M=self.M, p=self.p, F=self._daeic_F, J=self._daeic_J)
        try:
            with linsolver(self._daeic_backend):
                return DaeIc(proxy, y, t, self.opts.rtol), None
        # LinAlgError is a ValueError, so it is caught first
        except (np.linalg.LinAlgError, RuntimeError, StepFailure) as e:
            error = e
        except ValueError as e:
            if e.args != ('Need Better y0',):
                raise
            error = e
        return None, f"DaeIc found no consistent initial values ({type(error).__name__}: {error})"

    # -- the loop -----------------------------------------------------------

    def step(self):
        """Advance by one accepted step; ``False`` once the run is over.

        Between two calls, ``t`` and ``u`` are the end of the step just
        taken, and ``interp`` evaluates inside it.
        """
        if self.finished:
            return False
        while True:
            if not self.loopheader():
                return False
            if self.next_step_tstop and abs(self.dt) < math.ulp(abs(self.t)):
                self._skip_to_tstop()
            else:
                self.perform_step()
            self.loopfooter()
            if self.failed:
                return False
            if self.accept_step:
                self.policy.handle_tstop(self)
                self.finished = bool(self.terminated or self.policy.at_end(self))
                return not self.finished

    def solve(self):
        """Run to the end and return the ``daesol``."""
        while self.step():
            pass
        return self.postamble()

    def terminate(self):
        """End the run after the current callback, or before the next step."""
        self.terminated = True
        self.retcode = 'terminated'
        self.finished = True

    def model_modified(self):
        """Run the modification protocol after the caller changed the state or
        the model between two ``step()`` calls.

        The caller changes ``u``, an array of ``dae.p`` or ``dae.M.data``, or
        rebinds ``dae.p`` or ``dae.M``, and then calls this; it is the only
        supported way to do so. The state is made consistent at ``t`` with
        ``DaeIc``, whose failure ends the run. The change may have overwritten
        the end state of the step just taken, so ``interp`` accepts only
        ``tq == t`` until the next accepted step. Before the first step the
        changed state is the start of that step. After a failure there is
        nothing to modify and the call does nothing.
        """
        if self.failed:
            return
        if self.iter > 0:
            self._interp_valid = False
        self._modification_protocol()

    def loopheader(self):
        """Commit the accepted step or apply the rejection, then check and
        bound the step of the next attempt; ``False`` when the run fails."""
        if self.iter > 0:
            if self.accept_step:
                self.apply_step()
            elif not self.force_stepfail:
                self.controller.on_reject(self, self.q)
            # after a failed attempt loopfooter has already divided dt
        reason = self.policy.check_error(self)
        if reason is not None:
            self._fail(reason)
            return False
        self.iter += 1
        self.policy.fix_dt_at_bounds(self)
        self.policy.modify_dt_for_tstops(self)
        self.force_stepfail = False
        return True

    def apply_step(self):
        """The commit point: the accepted state becomes the start of the next step."""
        np.copyto(self.uprev, self.u)
        self.u_step = self.u
        self._interp_valid = True
        self.new_step = True
        self.dt = self.dtpropose

    def loopfooter(self):
        """Accept or reject the attempt; on acceptance advance ``t``, propose
        the next step and save."""
        opts = self.opts
        if self.force_stepfail:
            self.accept_step = False
            self.stats.nreject += 1
            self.nconsecutive_reject += 1
            if opts.adaptive:
                self.dt = self.dt / opts.failfactor
            else:
                self._fail('the step failed at a fixed step size: ' + self._stepfail_reason)
            return
        if self._skip_step:
            self.accept_step = True
        elif opts.adaptive:
            self.policy.sanitize_EEst(self)
            self.q = self.controller.stepsize(self)
            self.accept_step = self.controller.accepts(self)
        elif not self._all_finite(self.u):
            self._fail('the state is not finite')
            return
        else:
            self.accept_step = True
        if not self.accept_step:
            self.stats.nreject += 1
            self.nconsecutive_reject += 1
            return
        self.stats.nstep += 1
        self.nconsecutive_reject = 0
        self.dt_step = self.dt
        ttmp = self.t + self.dt
        self.tprev = self.t
        self.t = self.tstop_target if self.next_step_tstop else ttmp
        self.t_step = self.t
        if self._skip_step:
            self._skip_step = False
            self.dtpropose = self.dt_untruncated
        elif opts.adaptive:
            self.dtpropose = self.policy.dt_propose(self, self.controller.on_accept(self, self.q))
        else:
            self.dtpropose = opts.dt0
        self.next_step_tstop = False
        try:
            self.handle_callbacks()
        except StepFailure as e:
            # an arithmetic error of the model in addsteps or in an
            # interpolation: the step stands, and the run ends at its end
            self._fail(f"the model failed after the step was accepted: {e}")
            return
        if self._pbar is not None:
            self._pbar.update(self.t - self.tprev)

    def _skip_to_tstop(self):
        """An attempt shorter than one unit in the last place of ``t`` before a
        stop time: the state is kept and the step lands on the stop time."""
        np.copyto(self.u, self.uprev)
        self.EEst = 0.0
        self._interp_ready = False
        self._skip_step = True

    def handle_callbacks(self):
        """Handle the continuous and then the discrete callbacks of the
        accepted step, then save its rows unless a callback saved them."""
        self._te = None
        saved = False
        if self.continuous_callbacks:
            saved = self._apply_continuous_callbacks()
            if self.failed:
                return
        if self.discrete_callbacks:
            saved = self._apply_discrete_callbacks(saved)
            if self.failed:
                return
        if not saved:
            self.policy.savevalues(self)

    def _apply_continuous_callbacks(self):
        """Locate and handle the crossings of the accepted step; whether rows were saved.

        Without an acting crossing every recorded crossing is logged at its
        root with the interpolated state, and the step stands. Otherwise the
        earliest acting root ``te`` is one event instant for every callback:
        ``t`` and ``u`` move to ``te`` and the interpolated state there, while
        the step as computed stays available through ``u_step``, so that
        every later interpolation of the step is unchanged; the recorded
        crossings before ``te`` are logged, the rows up to ``te`` saved, and
        every component that crosses at ``te`` is handled in list order, its
        crossing logged with the state before any change, a terminal one
        ending the run and an ``affect`` receiving the components of its
        callback. After the affects, the modification protocol runs once. The
        next step is proposed with the full length of this one.
        """
        states = self.continuous_callbacks
        te, found = locate(self, states)
        if te is None:
            for c in found:
                self.events.record(c.root, self.interp(c.root), c.i)
            for st in states:
                st.g0 = st.g1
            return False
        u_e = self.interp(te)
        self._detach()
        self.t = te
        np.copyto(self.u, u_e)
        self._te = te
        at_te = {}
        for c in found:
            if c.root < te:
                self.events.record(c.root, self.interp(c.root), c.i)
            else:
                at_te.setdefault(c.st.order, []).append(c.i)
        self.policy.savevalues(self)
        handled = [(states[k], at_te[k]) for k in sorted(at_te)]
        if any(st.cb.save_positions[0] for st, _ in handled) and self.sol.last_t != te:
            self.sol.push(te, self.u.copy())
        affected = False
        for st, idx in handled:
            if st.cb.record:
                for i in idx:
                    self.events.record(te, u_e, i)
            if st.terminal[idx].any():
                self.terminate()
            if st.cb.affect is not None:
                if not affected:
                    self._freeze_step()
                    affected = True
                st.cb.affect(self, np.array(idx, dtype=np.int64))
        if affected and not self._modification_protocol(handled):
            return True
        if any(st.cb.save_positions[1] for st, _ in handled):
            self.sol.push(te, self.u.copy())
        self.dtpropose = self.dt_step
        return True

    def _apply_discrete_callbacks(self, saved):
        """Apply, in list order, every discrete callback whose condition holds
        at ``(t, u)``, unless the run is terminated; whether rows were saved.

        The rows up to ``t`` are saved once, before the first affect, unless
        the continuous callbacks, which ``saved`` reports, have saved them.
        Each callback then saves the state before its affect unless a row at
        ``t`` was saved already, runs its affect and the modification
        protocol, and saves the state after it; its ``save_positions`` selects
        these two saves. A callback that terminates the run ends the list.
        """
        t = self.t
        for cb in self.discrete_callbacks:
            if self.terminated:
                break
            self.stats.ncondition += 1
            if not cb.condition(t, self.u, self):
                continue
            if not saved:
                # once per step end: the legacy savevalues pushes (t, u)
                # without looking at the last saved row
                self.policy.savevalues(self)
                saved = True
            if cb.save_positions[0] and self.sol.last_t != t:
                self.sol.push(t, self.u.copy())
            self._freeze_step()
            cb.affect(self)
            if not self._modification_protocol():
                break
            if cb.save_positions[1]:
                self.sol.push(t, self.u.copy())
        return saved

    def _modification_protocol(self, fired=()):
        """Make the run consistent after the state or the model changed at
        ``t``; ``False`` when ``DaeIc`` fails, which ends the run.

        The model is read again from ``dae`` and ``model_epoch`` advances, so
        that every cache derived from ``M`` is rebuilt; the linear-solver
        cache returns to the state of a new call; ``DaeIc`` makes ``u``
        consistent; ``F0``, ``J0`` and ``dF/dt`` are evaluated anew at the
        next attempt and the algorithm clears its history; the bottom values
        of the continuous callbacks are taken at the changed state. ``fired``
        holds the components of each continuous callback that crossed at
        ``t``, which a ``'left'`` callback keeps for the repeat nudge of the
        next step. The next commit makes ``u`` the start of the next step.
        """
        self._detach()
        dae = self.dae
        self.M = dae.M
        self.p = dae.p
        self.model_epoch += 1
        self.linalg.reset()
        y, reason = self._daeic(self.u, self.t)
        if reason is not None:
            self._fail(reason)
            return False
        if y is not self.u:
            np.copyto(self.u, y)
        if self.iter == 0:
            # no step to commit: the changed state starts the first step
            np.copyto(self.uprev, self.u)
            self.u_step = self.u
        self.new_step = True
        self._nl_eta = 1.0
        self.alg.reset_history(self, self.cache)
        for st in self.continuous_callbacks:
            st.restart(self)
        for st, idx in fired:
            if st.cb.rootfind == 'left':
                st.fired = np.array(idx, dtype=np.int64)
                st.fired_t = self.t
        return True

    def _freeze_step(self):
        """Keep the accepted step as computed before an affect changes the
        state, the parameters or the mass matrix: its end state, and the data
        that ``addsteps`` computes from them for the interpolant."""
        self._detach()
        if not self._interp_ready:
            self.alg.addsteps(self, self.cache)
            self._interp_ready = True

    def _detach(self):
        """Keep the end state of the accepted step apart from ``u``, once per
        step, before the core changes ``u``."""
        if self.u_step is self.u:
            np.copyto(self._u_step_buf, self.u)
            self.u_step = self._u_step_buf

    def postamble(self):
        """Close the run and build its ``daesol``."""
        if self._pbar is not None:
            self._pbar.close()
        if self.terminated and self.sol.last_t != self.t:
            self.sol.push(self.t, self.u.copy())
        if self.retcode is None:
            self.retcode = 'success'
        self.stats.ret = self.retcode
        self.stats.succeed = self.retcode != 'failed'
        return to_daesol(self.sol, self.events, self.stats, self._address)

    def _fail(self, reason):
        """End the run as failed and print one line; the rows saved so far are its result."""
        self.failed = True
        self.finished = True
        self.retcode = 'failed'
        self.stats.ret = 'failed'
        self.stats.succeed = False
        self.stats.t_fail = self.t
        print(f"{self.alg.scheme}: {reason} at t = {float(self.t)!r}; "
              f"the solution is returned up to t = {float(self.sol.last_t)!r}.")

    def _all_finite(self, u):
        fin = self._fin
        np.isfinite(u, out=fin)
        return fin.all()

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
        return self._interpolate((tq - self.tprev) / self.dt_step, out)

    def _interpolate(self, theta, out):
        """The algorithm's interpolant of the last accepted step at ``theta``,
        into ``out``, after its ``addsteps`` once per step."""
        alg = self.alg
        if not self._interp_ready:
            alg.addsteps(self, self.cache)
            self._interp_ready = True
        r = alg.interpolant(self, self.cache, theta, out)
        if r is not None and r is not out:
            np.copyto(out, r)
        return out
