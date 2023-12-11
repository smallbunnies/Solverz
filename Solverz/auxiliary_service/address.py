from __future__ import annotations

import numpy as np
from typing import Dict
from copy import deepcopy


def combine_Address(a1: Address, a2: Address) -> Address:
    a = Address()
    a.v = deepcopy(a1.v)
    var_len = a.total_size
    a.v.update(a2.v)
    for name in a2.v.keys():
        a.v[name] = a.v[name] + var_len
    return a


class Address:
    """
    Address objects for both equations and variables
    """

    def __init__(self):
        self.v: Dict[str, np.ndarray] = dict()

    def add(self, name: str, start: int = 0, end: int = 0):
        self.v[name] = np.arange(start, end + 1, dtype=int)

    def __getitem__(self, item):
        if isinstance(item, str):
            return slice(self.v[item][0], self.v[item][-1] + 1)

    def derive_alias(self, suffix: str):
        name = [var_name + suffix for var_name in self.v.keys()]
        a_dict = dict(zip(name, self.v.values()))
        a1 = Address()
        a1.v = a_dict
        return a1

    def update(self, name, start, end):
        self.v[name] = np.arange(start, end + 1, dtype=int)

    def __repr__(self):
        return self.v.__repr__()

    @property
    def size(self):
        return dict([(var_name, self.v[var_name][-1] - self.v[var_name][0] + 1) for var_name in self.v.keys()])

    @property
    def total_size(self):
        return np.sum(list(self.size.values()))
