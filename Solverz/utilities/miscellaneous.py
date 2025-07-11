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


def rearrange_list(A, B):
    """
    Rearranges list A based on the order specified in list B.

    Args:
        A (list): Original list of strings.
        B (list): List of strings defining the desired order.

    Returns:
        list: A new list with elements from A, reordered according to B.

    Raises:
        ValueError: If B contains duplicates or elements not found in A.
    """

    # Step 1: Check for duplicates in B
    if len(B) != len(set(B)):
        raise ValueError("B contains duplicate elements.")

    # Step 2: Check if all elements in B exist in A
    set_A = set(A)
    for item in B:
        if item not in set_A:
            raise ValueError(f"Element '{item}' not found in A.")

    # Step 3: Build a mapping: each element -> all its occurrences in A (preserve order)
    from collections import defaultdict
    mapping = defaultdict(list)
    for item in A:
        mapping[item].append(item)

    # Step 4: Collect elements from A that match the order in B
    result = []
    for key in B:
        result.extend(mapping[key])

    # Step 5: Collect elements in A that are not in B, preserving their original order
    not_in_B = set(B)
    remaining = [item for item in A if item not in not_in_B]

    # Step 6: Combine the two parts
    result.extend(remaining)

    return result
