import numpy as np
from scipy.sparse import csc_array


def numjac(F, t, y, Fty, thresh, S, g, vecon, central_diff):
    # Add t to the end of y and adjust thresh accordingly
    y = np.append(y, t)
    thresh = np.append(thresh, 1.0e-8)  # For df/dt

    facmax = 0.1
    ny = len(y)
    fac = np.sqrt(np.finfo(float).eps) + np.zeros(ny)
    yscale = np.maximum(0.1 * np.abs(y), thresh)
    del_ = (y + fac * yscale) - y
    jj = np.where(del_ == 0)[0]

    for j in jj:
        while True:
            if fac[j] < facmax:
                fac[j] = min(100 * fac[j], facmax)
                del_[j] = (y[j] + fac[j] * yscale[j]) - y[j]
                if del_[j] != 0:
                    break
            else:
                del_[j] = thresh[j]
                break

    if not vecon:
        nfevals = ny
        df = np.zeros((ny - 1, ny))

        for jj in range(0, ny):
            ydel = y.copy()
            ydel[jj] += del_[jj]
            t = ydel[-1]
            ydel = ydel[:-1]

            if central_diff:
                ydel_1 = y.copy()
                ydel_1[jj] -= del_[jj]
                t_1 = ydel_1[-1]
                ydel_1 = ydel_1[:-1]
                df[:, jj] = (F(t, ydel) - F(t_1, ydel_1)) / (2 * del_[jj])
                nfevals += 1
            else:
                df[:, jj] = (F(t, ydel) - Fty) / del_[jj]
    else:
        # Vectorized function F
        raise NotImplementedError("Vectorized function F handling is not implemented.")

    # Convert df to sparse matrix dFdyt
    dFdyt = csc_array(df)

    return Fty, dFdyt, nfevals


def numjac_ae(F, y, thresh):
    # Add t to the end of y and adjust thresh accordingly

    facmax = 0.1
    ny = len(y)
    fac = np.sqrt(np.finfo(float).eps) + np.zeros(ny)
    yscale = np.maximum(0.1 * np.abs(y), thresh)
    del_ = (y + fac * yscale) - y
    jj = np.where(del_ == 0)[0]

    for j in jj:
        while True:
            if fac[j] < facmax:
                fac[j] = min(100 * fac[j], facmax)
                del_[j] = (y[j] + fac[j] * yscale[j]) - y[j]
                if del_[j] != 0:
                    break
            else:
                del_[j] = thresh[j]
                break

    nfevals = ny
    df = np.zeros((ny, ny))

    for jj in range(0, ny):
        ydel = y.copy()
        ydel[jj] += del_[jj]

        ydel_1 = y.copy()
        ydel_1[jj] -= del_[jj]
        df[:, jj] = (F(ydel) - F(ydel_1)) / (2 * del_[jj])
        nfevals += 1

    # Convert df to sparse matrix dFdyt
    dFdyt = csc_array(df)

    return dFdyt, nfevals
