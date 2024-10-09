from __future__ import annotations

import numpy as np
from typing import Dict
from copy import deepcopy
from .type_checker import is_integer


def combine_Address(a1: Address, a2: Address) -> Address:
    a = Address()
    a.length_array = np.concatenate([a1.length_array, a2.length_array])
    a.object_list = a1.object_list + a2.object_list
    a.update_v_cache()
    return a


class Address:
    """
    Address objects for both equations and variables
    """

    def __init__(self):
        self.length_array = np.zeros((0,), dtype=int)
        self.object_list = []
        self.v_cache: Dict[str, np.ndarray] = dict()

    def add(self, name: str, length: int = 0):
        if name not in self.object_list:
            self.object_list.append(name)
            self.length_array = np.append(self.length_array, int(length))
            self.update_v_cache()
        else:
            raise KeyError(f"Variable {name} already exists!")

    def __getitem__(self, item):
        if isinstance(item, str):
            return slice(self.v[item][0], self.v[item][-1] + 1)

    def derive_alias(self, suffix: str):
        a1 = Address()
        a1.object_list = [var_name + suffix for var_name in self.object_list]
        a1.length_array = self.length_array
        a1.update_v_cache()
        return a1

    def update(self, name, length):
        if name not in self.v.keys():
            raise KeyError(f"Non-existent name {name}, use Address.add() instead")
        self.length_array[self.object_list.index(name)] = length
        self.update_v_cache()

    def update_v_cache(self):
        address_list = []
        for name in self.object_list:
            idx = self.object_list.index(name)
            start = np.sum(self.length_array[0:idx])
            address_list.append(np.arange(start, start + self.length_array[idx], dtype=int))
        self.v_cache = dict(zip(self.object_list, address_list))

    def inquiry_eqn_name(self, addr: int):
        """
        Given eqn address (number), find the equation name.
        """
        if not is_integer(addr):
            raise ValueError(f"Address should be integer but {addr}!")
        if addr < 0:
            raise ValueError(f"No negative address allowed!")

        current_sum = -1 # The address should start from 0
        for i, value in enumerate(self.length_array):
            current_sum += value
            if current_sum >= addr:
                break
        if addr > current_sum:
            raise ValueError(f"Input address bigger than maximum address {current_sum}!")
        eqn_name = self.object_list[i]
        if addr in self.v[eqn_name].tolist():
            return eqn_name
        else:
            raise ValueError(f"How could this happen?")

    @property
    def v(self) -> Dict[str, np.ndarray]:
        return self.v_cache

    def __repr__(self):
        return self.v.__repr__()

    @property
    def size(self):
        return dict(zip(self.object_list, self.length_array))

    @property
    def total_size(self):
        return np.sum(self.length_array)

    def __eq__(self, other):
        if not isinstance(other, Address):
            return False
        if self.object_list != other.object_list:
            return False
        if not np.all(np.isclose(self.length_array, other.length_array)):
            return False
        return True
