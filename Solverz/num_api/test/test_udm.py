"""
Test the user defined modules.
"""

import importlib
import os
import re
import shutil
from pathlib import Path

import pytest

from Solverz.num_api.user_function_parser import add_my_module, reset_my_module_paths

mymodule_code = """import numpy as np
from numba import njit


@njit(cache=True)
def c(x, y):
    x = np.asarray(x).reshape((-1,))
    y = np.asarray(y).reshape((-1,))

    z = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] <= y[i]:
            z[i] = x[i]
        else:
            z[i] = y[i]
    return z
"""


def test_udm():
    # Create a .Solverz_test_temp directory in the user's home directory
    user_home = str(Path.home())
    solverz_dir = os.path.join(user_home, '.Solverz_test_temp')

    # Create the .Solverz directory if it does not exist
    if not os.path.exists(solverz_dir):
        os.makedirs(solverz_dir)

    file_path = os.path.join(solverz_dir, r'your_module.py')
    file_path1 = os.path.join(solverz_dir, r'fake1.jl')

    # Write the new paths to the file, but only if they are not already present
    with open(file_path, 'a') as file:
        file.write(mymodule_code)

    with open(file_path1, 'a') as file:
        file.write(mymodule_code)

    with pytest.raises(ValueError,
                       match=re.escape(f"The path {solverz_dir} is not a file.")):
        add_my_module([solverz_dir])

    with pytest.raises(ValueError,
                       match=re.escape(f"The path {os.path.join(user_home, '.Solverz_test_temp1')} does not exist.")):
        add_my_module([os.path.join(user_home, '.Solverz_test_temp1')])

    with pytest.raises(ValueError,
                       match=re.escape(f"The file {file_path1} is not a Python file.")):
        add_my_module([file_path1])

    add_my_module([file_path])

    import Solverz
    importlib.reload(Solverz.num_api.module_parser)
    from Solverz.num_api.module_parser import your_module
    import numpy as np
    np.testing.assert_allclose(your_module.c(np.array([1, 0]), np.array([2, -1])), np.array([1, -1]))

    shutil.rmtree(solverz_dir)
    reset_my_module_paths()
