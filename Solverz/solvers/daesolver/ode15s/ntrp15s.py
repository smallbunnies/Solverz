from ..utilities import *


def ntrp15s(tinterp, tnew, ynew, dt, dif, k):
    s = (tinterp - tnew) / dt

    if k == 1:
        yinterp = ynew + dif[:, 0] * s
    else:
        kI = np.arange(1, k + 1).reshape((-1, 1))
        yinterp = ynew + (dif[:, 0:k] @ np.cumprod((s + kI - 1) / kI, axis=0)).reshape(-1)

    return yinterp
