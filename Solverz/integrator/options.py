"""The options of one integration, read once from ``Opt``."""
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from Solverz.solvers.laesolver import resolve_backend
from Solverz.solvers.option import Opt

__all__ = ['IntegratorOptions']


def _frozen(x):
    """A scalar unchanged; an array as a read-only float64 copy, so that a
    later write to the caller's array cannot reach a running integration.

    A 0-d array is copied with its dtype, which the arithmetic of legacy
    Rodas sees, and made read-only as well.
    """
    if isinstance(x, np.ndarray) and x.ndim == 0:
        a = x.copy()
        a.flags.writeable = False
        return a
    if np.ndim(x) == 0:
        return x
    a = np.array(x, dtype=np.float64)
    a.flags.writeable = False
    return a


@dataclass(frozen=True, eq=False)
class IntegratorOptions:
    """The options of one integration.

    ``from_opt`` reads ``Opt`` once and never writes to it. The names are
    those of the integrator: ``safety`` is ``opt.f_savety``, ``qmin`` is
    ``opt.fac1``, ``qmax`` is ``opt.fac2``, ``qmax_init`` is ``opt.facmax``,
    ``dt0`` is ``opt.hinit`` and ``dtmax`` is ``opt.hmax``, or
    ``|tend - t0|`` when that is ``None``. ``saveat`` holds the nodes
    ``tspan[1:]`` when ``dense``, that is, when ``tspan`` has more than two
    entries.

    In the legacy-compatible configuration ``t0``, ``tend``, ``saveat``,
    ``dt0`` and ``dtmax`` keep the types legacy Rodas computes them with, so
    that an integer ``tspan`` gives integer times as it does there; in the
    default configuration they are floats.
    """

    t0: Any
    tend: Any
    saveat: Optional[np.ndarray]
    dense: bool
    rtol: Any
    atol: Any
    safety: Any
    qmin: Any
    qmax: Any
    qmax_init: Any
    dt0: Any
    dtmax: Any
    adaptive: bool
    legacy_compat: bool
    linsolver: str
    event: Optional[Callable]
    pbar: bool
    failfactor: float = 2.0
    max_consecutive_reject: int = 100

    @classmethod
    def from_opt(cls, opt, alg, tspan):
        """Read ``opt`` for the algorithm ``alg`` on ``tspan``.

        Raises ``ValueError`` for ``t0 > tend``, for a ``hinit`` or a
        ``hmax`` that is not ``None`` and not positive, for a ``tspan`` of
        more than two entries that does not increase strictly in the default
        configuration, and for a run with a fixed step and no ``hinit``.
        """
        if opt is None:
            opt = Opt()
        legacy = bool(alg.legacy_compat)
        if legacy:
            ts = np.array(tspan)
        else:
            ts = np.array(tspan, dtype=np.float64)
        ts.flags.writeable = False
        t0, tend = ts[0], ts[-1]
        if t0 > tend:
            raise ValueError(f't0: {t0} > tend: {tend}')
        hinit = opt.hinit
        if hinit is not None and hinit <= 0:
            raise ValueError(f"opt.hinit = {hinit!r} is not positive")
        # a step bounded by hmax <= 0 is raised to the smallest step at every
        # attempt, and the run would advance by 16 ulp per step without end
        if opt.hmax is not None and not opt.hmax > 0:
            raise ValueError(f"opt.hmax = {opt.hmax!r} is not positive")
        dense = len(ts) > 2
        if dense and not legacy and not np.all(ts[1:] > ts[:-1]):
            raise ValueError("a tspan of more than two entries must increase strictly")
        adaptive = bool(alg.adaptive) and not opt.fix_h
        if not adaptive and hinit is None:
            if not alg.adaptive:
                raise ValueError(
                    f"{alg.scheme} has no error estimate (adaptive = False) and runs with the "
                    f"fixed step opt.hinit, which is not set; an algorithm whose perform_step "
                    f"returns an error estimate declares adaptive = True")
            raise ValueError("opt.fix_h needs opt.hinit")
        dtmax = np.abs(tend - t0) if opt.hmax is None else opt.hmax
        if not legacy:
            t0, tend, dtmax = float(t0), float(tend), float(dtmax)
            if hinit is not None:
                hinit = float(hinit)
        return cls(t0=t0,
                   tend=tend,
                   saveat=ts[1:] if dense else None,
                   dense=dense,
                   rtol=_frozen(opt.rtol),
                   atol=_frozen(opt.atol),
                   safety=opt.f_savety,
                   qmin=opt.fac1,
                   qmax=opt.fac2,
                   qmax_init=opt.facmax,
                   dt0=hinit,
                   dtmax=dtmax,
                   adaptive=adaptive,
                   legacy_compat=legacy,
                   linsolver=resolve_backend(getattr(opt, 'linsolver', None)),
                   event=opt.event,
                   pbar=opt.pbar)
