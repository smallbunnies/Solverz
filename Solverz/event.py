from typing import Union, List, Tuple, Dict
from numbers import Number

import numpy as np


class Event:

    def __init__(self,
                 name: str = None,
                 time: Union[List[Number], np.ndarray] = None):
        self.name = name
        self.time = time
        self.index: Dict[str, Tuple[int]] = dict()
        self.var_value: Dict[str, np.ndarray] = dict()

    def interpolate(self,
                    var,
                    t):
        # Fixme: 1-d参数 2-d参数
        return np.interp(t, self.time, self.var_value[var])

    def add_var(self,
                var: str,
                v,
                index):
        v = np.asarray(v, dtype=np.float64)
        self.var_value[var] = v
        self.index[var] = index

    def __repr__(self):
        return f'Simulation Event: +{self.name}'
