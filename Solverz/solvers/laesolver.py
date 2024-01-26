from typing import Union

import numpy as np
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, linalg as sla


def solve(A, b):
    if isinstance(A, (csc_array, csc_matrix, csr_array, csr_matrix)):
        return sla.spsolve(A, b)
    else:
        return np.linalg.solve(A, b)


def lu_decomposition(A: Union[np.ndarray, csc_array, csc_matrix]):
    if isinstance(A, np.ndarray):
        return dense_decomposition(A)
    else:
        return sp_decomposition(A)


class dense_decomposition:
    def __init__(self,
                 A: np.ndarray):
        self.A = A

    def solve(self, b):
        return solve(self.A, b)


class sp_decomposition:
    def __init__(self,
                 A: Union[(csc_array, csc_matrix)]):
        self.lu_solve = sla.factorized(A)

    def solve(self, b):
        return self.lu_solve(b)
