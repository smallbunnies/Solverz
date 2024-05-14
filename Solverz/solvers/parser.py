from typing import Callable, Any, List
import functools
import numpy as np
from Solverz.variable.variables import Vars, TimeVars
from Solverz.utilities.address import Address
from Solverz.num_api.num_eqn import nAE, nDAE, nFDAE
from Solverz.solvers.option import Opt
from Solverz.solvers.solution import aesol, daesol


def ae_io_parser(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    The parser is used to:
     1. Make input/output type consistent.
     2. Convert Vars input to np.ndarray.
    """

    @functools.wraps(func)
    def wrapper(eqn: nAE, y0: np.ndarray | Vars, opt: Opt = None):
        # Convert Vars input to np.ndarray if necessary
        original_y0_is_vars = isinstance(y0, Vars)
        if original_y0_is_vars:
            y = y0.array  # Convert y0 from Vars to np.ndarray
        else:
            y = y0

        # Dispatch AE solvers and capture results
        sol = func(eqn, y, opt)

        # Wrap the output in Vars if the original y0 was a Vars instance
        if original_y0_is_vars:
            sol.y = parse_ae_v(sol.y, y0.a)  # Assume y0.a is accessible and relevant

        # Return results, with stats if they were provided
        return sol

    return wrapper


def fdae_io_parser(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    The parser is used to:
     1. Make input/output type consistent.
     2. Convert Vars input to np.ndarray.
    """

    @functools.wraps(func)
    def wrapper(eqn: nFDAE, tspan: List | np.ndarray, y0: np.ndarray | Vars, opt: Opt = None):
        # Convert Vars input to np.ndarray if necessary
        original_y0_is_vars = isinstance(y0, Vars)
        if original_y0_is_vars:
            y = y0.array  # Convert y0 from Vars to np.ndarray
        else:
            y = y0

        # Dispatch AE solvers and capture results
        sol = func(eqn, tspan, y, opt)

        # Wrap the output in Vars if the original y0 was a Vars instance
        if original_y0_is_vars:
            sol.Y = parse_dae_v(sol.Y, y0.a)  # Assume y0.a is accessible and relevant

        # Return results, with stats if they were provided
        return sol

    return wrapper


def dae_io_parser(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    The parser is used to:
     1. Make input/output type consistent.
     2. Convert Vars input to np.ndarray.
    """

    @functools.wraps(func)
    def wrapper(eqn: nDAE, tspan: List | np.ndarray, y0: np.ndarray | Vars, opt: Opt = None):
        # Convert Vars input to np.ndarray if necessary
        original_y0_is_vars = isinstance(y0, Vars)
        if original_y0_is_vars:
            y = y0.array  # Convert y0 from Vars to np.ndarray
        else:
            y = y0

        # Dispatch AE solvers and capture results
        sol = func(eqn, tspan, y, opt)

        # Wrap the output in Vars if the original y0 was a Vars instance
        if original_y0_is_vars:
            sol.Y = parse_dae_v(sol.Y, y0.a)  # Assume y0.a is accessible and relevant
            if sol.te is not None:
                if sol.te.size > 0:
                    sol.ye = parse_dae_v(sol.ye, y0.a)
        # Return results, with stats if they were provided
        return sol

    return wrapper


def parse_ae_v(y: np.ndarray, var_address: Address):
    return Vars(var_address, y)


def parse_dae_v(y: np.ndarray, var_address: Address):
    temp = Vars(var_address, y[0, :])
    temp = TimeVars(temp, y.shape[0])
    temp.array[:, :] = y
    return temp
