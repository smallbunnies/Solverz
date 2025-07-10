from scipy.sparse import coo_array


def derive_incidence_matrix(indices, m, n) -> coo_array:
    """
    Convert indices to `m*n` incidence matrix
    """

    data = []
    row = []
    col = []
    for i in indices:
        data.append(1)
        row.append(indices.index(i))
        col.append(i)

    return coo_array((data, (row, col)), shape=(m, n), dtype=int)
