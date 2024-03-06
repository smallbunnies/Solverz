import numpy as np


class aesol:
    __slots__ = ['y', 'stats']

    def __init__(self,
                 y,
                 stats=None):
        self.y = y
        self.stats = stats


class daesol:
    __slots__ = ['T', 'Y', 'te', 'ie', 'ye', 'stats']

    def __init__(self,
                 T,
                 Y,
                 te=None,
                 ye=None,
                 ie=None,
                 stats=None):
        self.T = T
        self.Y = Y
        self.te = te
        self.ie = ie
        self.ye = ye
        self.stats = stats
