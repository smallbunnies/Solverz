import numpy as np
from solverz_array import SolverzArray


def derive_incidence_matrix(f_node: np.ndarray,
                            t_node: np.ndarray):
    num_node = np.max([np.max(f_node), np.max(t_node)])
    num_pipe = f_node.shape[0]
    temp = np.zeros((num_node, num_pipe))
    for pipe in range(1, num_pipe + 1):
        temp[f_node[pipe - 1] - 1, pipe - 1] = -1
        temp[t_node[pipe - 1] - 1, pipe - 1] = 1

    return SolverzArray(temp)


def derive_v_plus(v: SolverzArray):
    return (v + np.abs(v)) / 2


def derive_v_minus(v: SolverzArray):
    return (v - np.abs(v)) / 2
