"""Step-size control: after each attempt of an adaptive run, whether the
attempt is accepted and which step comes next.

A controller belongs to one Integrator and keeps its state itself; ``Opt``
is never written. ``stepsize`` runs after every adaptive attempt that did
not fail and returns the quantity ``q`` that ``on_accept`` or ``on_reject``
then receives. For ``IController`` and ``PIController`` ``q`` is the inverse
growth factor of OrdinaryDiffEq.jl, so the next step is ``dt / q``.

Every ``rodas.py:N`` refers to legacy Rodas at commit ``056e87a``, before its
deprecation warning moved its lines.
"""
import numpy as np

__all__ = ['Controller', 'IController', 'PIController', 'LegacyRodasController']


def _error_order(alg):
    """The order ``k`` of the error estimate, ``O(h**k)``; ``order`` when not stated."""
    k = alg.error_order
    return alg.order if k is None else k


class Controller:
    """The protocol of a step-size controller.

    ``stepsize(integ)`` reads ``integ.EEst`` and ``integ.dt`` and returns
    ``q``; ``accepts(integ)`` decides the attempt; ``on_accept(integ, q)``
    returns the step proposed for the next step; ``on_reject(integ, q)`` sets
    ``integ.dt`` for the retry.
    """

    def __init__(self, opts, alg):
        pass

    def stepsize(self, integ):
        raise NotImplementedError(f"{type(self).__name__} does not define stepsize")

    def accepts(self, integ):
        return integ.EEst <= 1.0

    def on_accept(self, integ, q):
        raise NotImplementedError(f"{type(self).__name__} does not define on_accept")

    def on_reject(self, integ, q):
        raise NotImplementedError(f"{type(self).__name__} does not define on_reject")


class IController(Controller):
    """Integral control: ``q = E**(1/k) / safety``, clamped to ``[1/qmax, 1/qmin]``,
    with ``k`` the algorithm's ``error_order``, and the next step ``dt / q``.

    After a rejection the growth cap is 1 until the next accepted attempt,
    as legacy Rodas keeps it, so no retry grows the step. The steady band
    ``[1, 1]`` keeps the step only when ``q`` is exactly 1.
    """

    qsteady_min = 1.0
    qsteady_max = 1.0

    def __init__(self, opts, alg):
        self.safety = opts.safety
        self.qmin = opts.qmin
        self.qmax = opts.qmax
        self.qmax_now = opts.qmax_init
        self.expo = 1 / _error_order(alg)

    def stepsize(self, integ):
        E = integ.EEst
        if E == 0:
            return 1.0 / self.qmax_now
        return max(1.0 / self.qmax_now, min(1.0 / self.qmin, E ** self.expo / self.safety))

    def on_accept(self, integ, q):
        if self.qsteady_min <= q <= self.qsteady_max:
            q = 1.0
        self.qmax_now = self.qmax
        return integ.dt / q

    def on_reject(self, integ, q):
        integ.dt = integ.dt / q
        self.qmax_now = 1.0


class PIController(Controller):
    """Proportional-integral control, ``q = E**beta1 / Eold**beta2 / safety``
    clamped as in ``IController``, with ``beta1 = 7/(10 k)``, ``beta2 = 2/(5 k)``
    and ``Eold`` the error of the last accepted attempt, at least ``1e-4``.

    The growth cap after a rejection and the steady band are those of
    ``IController``. No built-in algorithm selects it; an algorithm does so
    by returning it from ``controller``.
    """

    qsteady_min = 1.0
    qsteady_max = 1.0
    qoldinit = 1e-4

    def __init__(self, opts, alg):
        k = _error_order(alg)
        self.beta1 = 7 / (10 * k)
        self.beta2 = 2 / (5 * k)
        self.safety = opts.safety
        self.qmin = opts.qmin
        self.qmax = opts.qmax
        self.qmax_now = opts.qmax_init
        self.q11 = 1.0
        self.errold = self.qoldinit

    def stepsize(self, integ):
        E = integ.EEst
        if E == 0:
            return 1.0 / self.qmax_now
        self.q11 = E ** self.beta1
        q = self.q11 / self.errold ** self.beta2
        return max(1.0 / self.qmax_now, min(1.0 / self.qmin, q / self.safety))

    def on_accept(self, integ, q):
        if self.qsteady_min <= q <= self.qsteady_max:
            q = 1.0
        self.errold = max(integ.EEst, self.qoldinit)
        self.qmax_now = self.qmax
        return integ.dt / q

    def on_reject(self, integ, q):
        integ.dt = integ.dt / min(1.0 / self.qmin, self.q11 / self.safety)
        self.qmax_now = 1.0


class LegacyRodasController(Controller):
    """The step-size rule of legacy Rodas, ``rodas.py:213-216, :349-355``.

    Every expression is legacy's, with the same NumPy functions and the same
    operand order, so the steps are legacy's bit for bit. ``q`` is legacy's
    factor ``fac``, and the proposed step ``dtnew`` is kept for ``on_accept``
    and ``on_reject``, which clamp it to ``[hmin, hmax]``. The growth cap
    ``facmax`` starts at the value ``opt.facmax`` has at call entry and
    becomes ``fac2`` after an accepted attempt and 1 after a rejected one;
    legacy writes it into ``opt``, this controller keeps it. ``hmin`` is
    ``16 * spacing(t0)`` with ``t0`` in its ``tspan`` dtype, fixed for the
    call.

    ``integ.EEst`` is the legacy error before its floor of ``1e-6``, which is
    applied here; the floor is below 1, so ``EEst <= 1`` decides as the
    floored error does, and a ``NaN`` rejects in both.
    """

    def __init__(self, opts, alg):
        self.safety = opts.safety
        self.fac1 = opts.qmin
        self.fac2 = opts.qmax
        self.pord = alg.tableau.pord
        self.hmin = 16 * np.spacing(opts.t0)
        self.hmax = opts.dtmax
        self.facmax = opts.qmax_init
        self.dtnew = None

    def stepsize(self, integ):
        err = np.maximum(integ.EEst, 1.0e-6)
        fac = self.safety / (err ** (1 / self.pord))
        fac = np.minimum(self.facmax, np.maximum(self.fac1, fac))
        self.dtnew = integ.dt * fac
        return fac

    def on_accept(self, integ, q):
        self.facmax = self.fac2
        return np.min([self.hmax, np.max([self.hmin, self.dtnew])])

    def on_reject(self, integ, q):
        self.facmax = 1
        integ.dt = np.min([self.hmax, np.max([self.hmin, self.dtnew])])
