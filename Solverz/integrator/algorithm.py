"""The author contract: what an integration algorithm defines."""
import inspect
import weakref
from types import SimpleNamespace

import numpy as np

__all__ = ['Algorithm', 'StepFailure']


class StepFailure(Exception):
    """An attempt of a step failed.

    The core rejects the attempt: an adaptive run retries with half the step,
    and a run with a fixed step fails. The linear-algebra service raises it on
    a failed factorization or solve, ``implicit`` on a failed Newton
    iteration, and an algorithm may raise it itself.
    """


class Algorithm:
    """One integration method, written as how a single step is taken.

    The step is ``perform_step``, in one of two styles. With ``inplace =
    False``, the default, ``perform_step(self, s)`` receives the
    ``StepContext`` ``s`` and returns ``y`` or ``(y, err)``. With ``inplace =
    True``, ``perform_step(self, integ, cache)`` writes ``integ.u`` and, in an
    adaptive run, ``integ.EEst``. The core supplies step control, saving,
    events and the services of ``s``; the hooks below have one signature in
    both styles and describe the last accepted step through ``integ.tprev``,
    ``integ.uprev``, ``integ.t_step``, ``integ.u_step`` and ``integ.dt_step``,
    never through ``integ.u``, which an event may have replaced.

    Traits: ``scheme`` names the method in ``Stats`` and messages; ``order``
    is its order; the error estimate is ``O(h**error_order)``, which sets the
    controller exponent ``1/error_order``; ``interp_order`` is the order of
    the interpolant; ``adaptive`` is true when ``perform_step`` returns an
    error estimate; ``explicit`` is true when the method never solves with
    ``W``; ``norm`` is ``'rms'`` or ``'max'``.
    """

    scheme = None
    order = None
    error_order = None
    interp_order = 1
    adaptive = False
    explicit = False
    inplace = False
    norm = 'rms'
    legacy_compat = False

    def perform_step(self, s):
        """Take one attempt of a step; see the class docstring for the two styles."""
        raise NotImplementedError(f"{type(self).__name__} does not define perform_step")

    def alloc(self, integ):
        """Allocate the algorithm's buffers once per integration."""
        return SimpleNamespace()

    def interpolant(self, integ, cache, theta, out):
        """The state at ``tprev + theta * dt_step``, into ``out``.

        An override may write into ``out`` and return ``None``, or return an
        array, which the core copies into ``out``. The default is the linear
        interpolant between ``uprev`` and ``u_step``.
        """
        np.subtract(integ.u_step, integ.uprev, out=out)
        np.multiply(out, theta, out=out)
        np.add(integ.uprev, out, out=out)

    def addsteps(self, integ, cache):
        """Compute, once per step, what ``interpolant`` needs beyond the step itself."""

    def reset_history(self, integ, cache):
        """Clear every datum kept from earlier steps; the state or the model has changed."""

    def initial_dt(self, integ):
        """The first step: ``opts.dt0``, or ``1e-6 * (tend - t0)`` without one."""
        opts = integ.opts
        if opts.dt0 is not None:
            return opts.dt0
        return 1e-6 * (opts.tend - opts.t0)


# inspect.signature is slow and EventLoop starts one call per segment, so each
# class is inspected once per value of inplace: class -> (inplace, error or None).
_STYLE = weakref.WeakKeyDictionary()


def _style_error(alg, inplace):
    name = type(alg).__name__
    if type(alg).perform_step is Algorithm.perform_step:
        return (f"{name} does not define perform_step; an algorithm writes its step as "
                f"perform_step(self, s), or as perform_step(self, integ, cache) with "
                f"inplace = True")
    nparam = len(inspect.signature(alg.perform_step).parameters)
    expected = 2 if inplace else 1
    if nparam != expected:
        form = 'perform_step(self, integ, cache)' if inplace else 'perform_step(self, s)'
        return (f"{name}.perform_step takes {nparam} parameters after self, but "
                f"inplace = {inplace} needs {expected}: {form}")
    return None


def check_style(alg):
    """Raise ``TypeError`` unless ``alg`` defines ``perform_step`` with the
    signature of its ``inplace`` style."""
    cls = type(alg)
    inplace = bool(alg.inplace)
    checked = _STYLE.get(cls)
    if checked is None or checked[0] != inplace:
        checked = (inplace, _style_error(alg, inplace))
        _STYLE[cls] = checked
    if checked[1] is not None:
        raise TypeError(checked[1])


class StepContext:
    """What a formula-style ``perform_step(self, s)`` sees of the Integrator.

    It is built once per Integrator, and its members are named after the
    symbols of the formulas. ``t`` and ``h`` are the start and the step of the
    current attempt, and ``new_step`` is false on a retry of the same step.
    ``y0`` is a read-only view of ``integ.uprev`` that is valid during the
    attempt only; an algorithm that keeps it copies it. ``M`` is never
    modified.
    """

    __slots__ = ('_integ', 'n', 'y0')

    def __init__(self, integ):
        self._integ = integ
        self.n = integ.n
        y0 = integ.uprev.view()
        y0.flags.writeable = False
        self.y0 = y0

    @property
    def t(self):
        return self._integ.t

    @property
    def h(self):
        return self._integ.dt

    @property
    def new_step(self):
        return self._integ.new_step

    @property
    def M(self):
        return self._integ.M

    @property
    def p(self):
        return self._integ.p

    @property
    def rtol(self):
        return self._integ.opts.rtol

    @property
    def atol(self):
        return self._integ.opts.atol

    @property
    def adaptive(self):
        return self._integ.opts.adaptive

    @property
    def cache(self):
        return self._integ.cache
