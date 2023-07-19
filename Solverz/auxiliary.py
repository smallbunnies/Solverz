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
            return self.v[item]

    def update(self, name, start, end):
        self.v[name] = np.arange(start, end+1, dtype=int)

    def __repr__(self):
        return self.v.__repr__()
