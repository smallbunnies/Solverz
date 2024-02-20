from typing import Callable, Any, List
import numpy as np
from Solverz.variable.variables import Vars, TimeVars
from Solverz.utilities.address import Address
from Solverz.num_api.num_eqn import nAE, nDAE, nFDAE
from Solverz.solvers.option import Opt


def ae_io_parser(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    The parser is used to:
     1. Make input/output type consistent.
     2. Convert Vars input to np.ndarray.
    """

    def wrapper(eqn: nAE, y0: np.ndarray | Vars, opt: Opt = None):
        # Convert Vars input to np.ndarray if necessary
        original_y0_is_vars = isinstance(y0, Vars)
        if original_y0_is_vars:
            y = y0.array  # Convert y0 from Vars to np.ndarray
        else:
            y = y0

        # Dispatch AE solvers and capture results
        results = func(eqn, y, opt)

        # If results include stats, unpack them
        if isinstance(results, tuple):
            y, stats = results
        else:
            y = results
            stats = None  # Define stats as None if not returned by func

        # Wrap the output in Vars if the original y0 was a Vars instance
        if original_y0_is_vars:
            y = parse_ae_v(y, y0.a)  # Assume y0.a is accessible and relevant

        # Return results, with stats if they were provided
        return (y, stats) if stats is not None else y

    return wrapper


def fdae_io_parser(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    The parser is used to:
     1. Make input/output type consistent.
     2. Convert Vars input to np.ndarray.
    """

    def wrapper(eqn: nFDAE, tspan: List | np.ndarray, y0: np.ndarray | Vars, opt: Opt = None):
        # Convert Vars input to np.ndarray if necessary
        original_y0_is_vars = isinstance(y0, Vars)
        if original_y0_is_vars:
            y = y0.array  # Convert y0 from Vars to np.ndarray
        else:
            y = y0

        # Dispatch AE solvers and capture results
        T, y, stats = func(eqn, tspan, y, opt)

        # Wrap the output in Vars if the original y0 was a Vars instance
        if original_y0_is_vars:
            y = parse_dae_v(y, y0.a)  # Assume y0.a is accessible and relevant

        # Return results, with stats if they were provided
        return T, y, stats

    return wrapper


def dae_io_parser(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    The parser is used to:
     1. Make input/output type consistent.
     2. Convert Vars input to np.ndarray.
    """

    def wrapper(eqn: nDAE, tspan: List | np.ndarray, y0: np.ndarray | Vars, opt: Opt = None):
        # Convert Vars input to np.ndarray if necessary
        original_y0_is_vars = isinstance(y0, Vars)
        if original_y0_is_vars:
            y = y0.array  # Convert y0 from Vars to np.ndarray
        else:
            y = y0

        # Dispatch AE solvers and capture results
        T, y, stats = func(eqn, tspan, y, opt)

        # Wrap the output in Vars if the original y0 was a Vars instance
        if original_y0_is_vars:
            y = parse_dae_v(y, y0.a)  # Assume y0.a is accessible and relevant

        # Return results, with stats if they were provided
        return T, y, stats

    return wrapper


def parse_ae_v(y: np.ndarray, var_address: Address):
    return Vars(var_address, y)


def parse_dae_v(y: np.ndarray, var_address: Address):
    temp = Vars(var_address, y[0, :])
    temp = TimeVars(temp, y.shape[0])
    temp.array[:, :] = y
    return temp
