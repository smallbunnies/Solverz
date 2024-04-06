import dill
import numpy as np
import pandas as pd
from Solverz.variable.variables import TimeVars
from Solverz.solvers.solution import aesol, daesol


def save(obj, filename: str):
    with open(filename, 'wb') as file:
        dill.dump(obj, file)


def load(filename: str):
    with open(filename, 'rb') as file:
        return dill.load(file)


def save_result(sol: aesol | daesol, name: str):
    if isinstance(sol, daesol):
        T = sol.T
        Y = sol.Y
        T = pd.DataFrame({'T': T})
        Y_dict = dict()
        for var in Y.var_list:
            tempd = dict()
            for j in range(Y.var_size[var]):
                tempd[var + f'_{j}'] = Y[var][:, j]
            Y_dict[var] = pd.DataFrame(tempd)
        with pd.ExcelWriter(f'{name}.xlsx', engine='openpyxl') as writer:
            # Write each DataFrame to a different sheet
            T.to_excel(writer, sheet_name='Time')
            for name, df in Y_dict.items():
                df.to_excel(writer, sheet_name=name)
    elif isinstance(sol, aesol):
        y = sol.y
        y_dict = dict()
        for var in y.var_list:
            y_dict[var] = pd.DataFrame({var: y[var]})
        with pd.ExcelWriter(f'{name}.xlsx', engine='openpyxl') as writer:
            # Write each DataFrame to a different sheet
            for name, df in y_dict.items():
                df.to_excel(writer, sheet_name=name)
    else:
        raise TypeError(f"Unknown solution type {type(sol)}")


