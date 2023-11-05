import numpy as np


class Address:
    """
    Address objects for both equations and variables
    """

    def __init__(self):
        self.v = dict()

    def add(self, name: str, start: int = 0, end: int = 0):
        self.v[name] = np.arange(start, end+1, dtype=int)

    def __getitem__(self, item):
        if isinstance(item, str):
            return slice(self.v[item][0], self.v[item][-1]+1)

    def length(self, name: str):
        return self.v[name][-1]-self.v[name][0]+1

    def update(self, name, start, end):
        self.v[name] = np.arange(start, end+1, dtype=int)

    def __repr__(self):
        return self.v.__repr__()

    @property
    def size(self):
        return dict([(var_name, self.length(var_name)) for var_name in self.v.keys()])

    @property
    def total_size(self):
        return np.sum(list(self.size.values()))
