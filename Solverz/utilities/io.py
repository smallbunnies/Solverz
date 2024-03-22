import dill
import numpy as np
import pandas as pd
from Solverz.variable.variables import TimeVars


def save(obj, filename: str):
    with open(filename, 'wb') as file:
        dill.dump(obj, file)


def load(filename: str):
    with open(filename, 'rb') as file:
        return dill.load(file)


def save_result(T: np.ndarray, Y: TimeVars, name: str):
    T = pd.DataFrame({'T': T})
    Y_dict = dict()
    for var in Y.var_list:
        tempd = dict()
        for j in range(Y.var_size[var]):
            tempd[var+f'{j}'] = Y[var][:, j]
        Y_dict[var] = pd.DataFrame(tempd)
    with pd.ExcelWriter(f'{name}.xlsx', engine='openpyxl') as writer:
        # Write each DataFrame to a different sheet
        T.to_excel(writer, sheet_name='Time')
        for name, df in Y_dict.items():
            df.to_excel(writer, sheet_name=name)
