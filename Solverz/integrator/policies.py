"""The two configurations of an integration, selected once per call.

``LegacyRodasPolicy`` reproduces legacy Rodas and ``DefaultPolicy`` follows
the rules of the integrator core. The Integrator builds the one that
``alg.legacy_compat`` selects, so that its loop contains no branch on the
configuration. Each policy names its ``dF/dt`` policy of ``derivative.py``
and holds its error norm, a function ``error_norm(integ, e)`` that works in
the Integrator's buffers ``_w``, ``_w2`` and ``_fin`` and returns an
``np.float64``.
"""
import numpy as np

__all__ = []


def legacy_error_norm(integ, e):
    """``max |e / (atol + rtol |u|)|``, the error of legacy Rodas, and
    ``1e6`` when ``u`` is not finite.

    A ``NaN`` from ``0/0``, which ``atol = 0`` and a zero component give with
    a finite ``u``, stays ``NaN`` and rejects the attempt, as in legacy.
    """
    u, w, opts = integ.u, integ._w, integ.opts
    np.abs(u, out=w)
    np.multiply(w, opts.rtol, out=w)
    np.add(w, opts.atol, out=w)
    np.divide(e, w, out=w)
    np.abs(w, out=w)
    err = np.max(w)
    fin = integ._fin
    np.isfinite(u, out=fin)
    # for float64, "not every entry is finite" is legacy's "some entry is
    # inf or NaN", without its two temporary arrays
    if not fin.all():
        err = np.float64(1.0e6)
    return err


def _scaled_error(integ, e):
    """``e / (atol + rtol max(|u|, |uprev|))`` into ``integ._w``."""
    w, w2, opts = integ._w, integ._w2, integ.opts
    np.abs(integ.u, out=w)
    np.abs(integ.uprev, out=w2)
    np.maximum(w, w2, out=w)
    np.multiply(w, opts.rtol, out=w)
    np.add(w, opts.atol, out=w)
    np.divide(e, w, out=w)
    return w


def max_error_norm(integ, e):
    """The largest scaled error component."""
    w = _scaled_error(integ, e)
    return np.max(np.abs(w, out=w))


def rms_error_norm(integ, e):
    """The root mean square of the scaled error components."""
    w = _scaled_error(integ, e)
    return np.sqrt(np.mean(np.square(w, out=w)))


class LegacyRodasPolicy:
    """The legacy-compatible configuration: legacy Rodas' ``dF/dt`` and error
    norm, the latter for every algorithm."""

    dfdt = 'legacy'

    def __init__(self, opts, alg):
        self.error_norm = legacy_error_norm


class DefaultPolicy:
    """The default configuration: the ``dF/dt`` of ode23s and the error scaled
    by ``max(|u|, |uprev|)``, in the norm the algorithm declares."""

    dfdt = 'ode23s'

    def __init__(self, opts, alg):
        if alg.norm == 'max':
            self.error_norm = max_error_norm
        elif alg.norm == 'rms':
            self.error_norm = rms_error_norm
        else:
            raise ValueError(f"{alg.scheme}.norm is {alg.norm!r}; it must be 'rms' or 'max'")
