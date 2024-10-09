from __future__ import annotations

import numpy as np

from Solverz.utilities.type_checker import is_number
from Solverz.solvers.stats import Stats


class aesol:
    __slots__ = ['y', 'stats']

    def __init__(self,
                 y,
                 stats=None):
        self.y = y
        self.stats = stats

    def __repr__(self):
        return f'ae solution using {self.stats.scheme}, succeed: {self.stats.succeed}'


class daesol:
    __slots__ = ['T', 'Y', 'te', 'ie', 'ye', 'stats']

    def __init__(self,
                 T=None,
                 Y=None,
                 te=None,
                 ye=None,
                 ie=None,
                 stats=None):
        self.T = T
        self.Y = Y
        self.te = te
        self.ie = ie
        self.ye = ye
        self.stats = Stats() if stats is None else stats

    def __getitem__(self, item):
        if isinstance(item, slice):
            T = self.T[item]
            Y = self.Y[item]
            stats = self.stats
            if self.ie is not None:
                event_idx = np.where(self.te >= T[0] & self.te <= T[-1])
                te = self.te[event_idx]
                ye = self.ye[event_idx]
                ie = self.ie[event_idx]
            else:
                te = None
                ye = None
                ie = None
            return daesol(T, Y, te, ye, ie, stats)

        elif is_number(item):
            item = int(item)
            return daesol(self.T[item], self.Y[item])
        else:
            raise NotImplementedError(f"Index type {type(item)} not implemented!")

    def append(self, sol: daesol):
        if self.stats.scheme is None:
            self.stats.scheme = sol.stats.scheme

        self.stats.nstep = self.stats.nstep + sol.stats.nstep
        self.stats.nfeval = self.stats.nfeval + sol.stats.nfeval
        self.stats.ndecomp = self.stats.ndecomp + sol.stats.ndecomp
        self.stats.nreject = self.stats.nreject + sol.stats.nreject

        self.T = np.concatenate([self.T, sol.T]) if self.T is not None else sol.T
        if self.Y is not None:
            self.Y.append(sol.Y)
        else:
            self.Y = sol.Y
        self.te = np.concatenate([self.te, sol.te]) if self.te is not None else sol.te
        if self.ye is not None:
            self.ye.append(sol.ye)
        else:
            self.ye = sol.ye
        self.ie = np.concatenate([self.ie, sol.ie]) if self.ie is not None else sol.ie
