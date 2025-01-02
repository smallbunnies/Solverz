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


def save_result(sol: aesol | daesol, name: str, dir: str = None):
    if dir is None:
        dir = ''
    else:
        dir = f'{dir}/'

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
        with pd.ExcelWriter(dir + f'{name}.xlsx', engine='openpyxl') as writer:
            # Write each DataFrame to a different sheet
            T.to_excel(writer, sheet_name='Time')
            for var, df in Y_dict.items():
                df.to_excel(writer, sheet_name=var)
        if sol.ie is not None:
            if len(sol.ie) > 0:
                te = pd.DataFrame({'te': sol.te})
                ie = pd.DataFrame({'ie': sol.ie})
                ye = sol.ye
                ye_d = dict()
                for var in ye.var_list:
                    tempd = dict()
                    for j in range(ye.var_size[var]):
                        tempd[var + f'_{j}'] = ye[var][:, j]
                    ye_d[var] = pd.DataFrame(tempd)
                with pd.ExcelWriter(dir + f'event_in_{name}.xlsx', engine='openpyxl') as writer:
                    # Write each DataFrame to a different sheet
                    te.to_excel(writer, sheet_name='Event Time')
                    ie.to_excel(writer, sheet_name='Event Index')
                    for var, df in ye_d.items():
                        df.to_excel(writer, sheet_name=var)
    elif isinstance(sol, aesol):
        y = sol.y
        y_dict = dict()
        for var in y.var_list:
            y_dict[var] = pd.DataFrame({var: y[var]})
        with pd.ExcelWriter(dir + f'{name}.xlsx', engine='openpyxl') as writer:
            # Write each DataFrame to a different sheet
            for name, df in y_dict.items():
                df.to_excel(writer, sheet_name=name)
    else:
        raise TypeError(f"Unknown solution type {type(sol)}")
