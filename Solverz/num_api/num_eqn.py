"""The numerical equation objects the solvers integrate.

Their residual ``F`` always accepts an ``out`` keyword, the array the
residual is written into: the in-place form of SciML's ``f!(du, u, p, t)``
in NumPy's ``out=`` spelling. A solver that owns its work arrays therefore
allocates nothing per residual evaluation, and a caller that passes no
``out`` receives a fresh array, so two residuals are never the same object.
The functions of ``made_numerical`` and of a rendered module accept ``out``
themselves; any other callable, such as a user's lambda or a module rendered
by an older Solverz, is given the keyword here by copying.
"""
import functools
import inspect
from typing import Callable, Dict


def _accepts_out(F: Callable) -> bool:
    """Whether ``F`` takes the residual array as a parameter named ``out``.

    A ``**kwargs`` catch-all does not count: it would accept the keyword and
    silently ignore it.
    """
    try:
        params = inspect.signature(getattr(F, 'py_func', F)).parameters
    except (TypeError, ValueError):
        return False
    return 'out' in params


def _with_out(F: Callable) -> Callable:
    """``F`` itself when it accepts ``out``; otherwise a wrapper that copies
    the out-of-place result into ``out`` when one is given."""
    if _accepts_out(F):
        return F

    @functools.wraps(F)
    def F_(*args, out=None):
        r = F(*args)
        if out is None:
            return r
        out[...] = r
        return out

    return F_


class nAE:

    def __init__(self,
                 F: Callable,
                 J: Callable,
                 p: Dict):
        self.F = _with_out(F)
        self.J = J
        self.p = p


class nFDAE:

    def __init__(self,
                 F: callable,
                 J: callable,
                 p: dict,
                 nstep: int = 0):
        self.F = _with_out(F)
        self.J = J
        self.p = p
        self.nstep = nstep


class nDAE:

    def __init__(self,
                 M,
                 F: Callable,
                 J: Callable,
                 p: Dict):
        self.M = M
        self.F = _with_out(F)
        self.J = J
        self.p = p
