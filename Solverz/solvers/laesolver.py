from typing import Union

import numpy as np
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, linalg as sla

# from scikits import umfpack
# umfpack is slow compared with superlu on Apple M4 with MACOS 15.6.1, scikits-umfpack 0.4.2, and suite-sparse 7.10.3.
# Also, it was found that umfpack was not accurate enough, causing non-convergence issues.

splu = sla.splu

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
        self.splu = splu(A)
        self.perm_r = self.splu.perm_r
        self.perm_c = self.splu.perm_c
        self.L = self.splu.L
        self.U = self.splu.U
        self.nnz = self.splu.nnz

    def solve(self, b):
        return self.splu.solve(b)
