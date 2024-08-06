from typing import Callable


class Opt:
    def __init__(self,
                 atol=1e-6,
                 rtol=1e-3,
                 f_savety=0.9,
                 facmax=6,
                 fac1=0.2,
                 fac2=6,
                 scheme='rodas4',
                 ite_tol=1e-5,
                 fix_h: bool = False,
                 hinit=None,
                 hmax=None,
                 pbar=False,
                 stats=False,
                 step_size=1e-3,
                 event: Callable = None,
                 event_duration=1e-8,
                 partial_decompose=False):
        self.atol = atol
        self.rtol = rtol
        self.f_savety = f_savety
        self.facmax = facmax
        self.fac1 = fac1
        self.fac2 = fac2
        self.fix_h = fix_h  # To force the step sizes to be invariant. This is not robust.
        self.hinit = hinit
        self.hmax = hmax
        self.scheme = scheme
        self.pbar = pbar
        self.ite_tol = ite_tol  # tol for iterative solver
        self.stats = stats
        self.step_size = step_size
        self.event = event
        self.event_duration = event_duration
        self.partial_decompose = partial_decompose
