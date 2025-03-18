import warnings

import numpy as np
from Solverz.solvers.daesolver.utilities import *
from .ntrp15s import ntrp15s


maxk = 5
G = np.array([1, 3 / 2, 11 / 6, 25 / 12, 137 / 60])
alpha = np.array([-37 / 200, -1 / 9, -0.0823, -0.0415, 0])
invGa = 1 / (G * (1 - alpha))
erconst = alpha * G + 1 / np.arange(2, 7)
difU = np.array([
    [-1, -2, -3, -4, -5],  # difU is its own inverse!
    [0, 1, 3, 6, 10],
    [0, 0, -1, -4, -10],
    [0, 0, 0, 1, 5],
    [0, 0, 0, 0, -1]
])
kJ = np.arange(1, maxk + 1).reshape((1, -1))
kI = kJ.T
difU = difU[0:maxk + 1, 0:maxk + 1]


@dae_io_parser
def ode15s(dae: nDAE,
           tspan: List | np.ndarray,
           y0: np.ndarray,
           opt: Opt = None):
    """
    The python implementation of MATLAB Ode15s without event detection [1]_.

    Parameters
    ==========

    dae : nDAE
        Numerical DAE object.

    tspan : List | np.ndarray
        Either
        - a list specifying [t0, tend], or
        - a `np.ndarray` specifying the time nodes that you are concerned about

    y0 : np.ndarray
        The initial values of variables

    opt : Opt
        The solver options, including:

        - rtol: 1e-3(default)|float
            The relative error tolerance
        - atol: 1e-6(default)|float
            The absolute error tolerance
        - hinit: None(default)|float
            Initial step size
        - hmax: None(default)|float
            Maximum step size.
        - pbar: False(default)|bool
            To display progress bar

    Returns
    =======

    sol : daesol
        The daesol object.


    References
    ==========

    .. [1] L. F. Shampine and M. W. Reichelt, “The MATLAB ODE Suite,” SIAM J. Sci. Comput., vol. 18, no. 1, pp. 1–22, Jan. 1997, doi: 10.1137/S1064827594276424.

    """
    if opt is None:
        opt = Opt()
    stats = Stats('ode15s')

    vsize = y0.shape[0]
    tspan = np.array(tspan)
    tend = tspan[-1]
    t0 = tspan[0]
    if t0 > tend:
        raise ValueError(f't0: {t0} > tend: {tend}')
    if opt.hmax is None:
        opt.hmax = np.abs(tend - t0)
    nt = 0
    t = t0
    hmin = 16 * np.spacing(t0)
    uround = np.spacing(1.0)

    if opt.pbar:
        pbar = tqdm(total=tend - t0)

    dense_output = False
    n_tspan = len(tspan)
    told = t0
    if n_tspan > 2:
        dense_output = True
        inext = 1
        tnext = tspan[inext]
    else:
        n_tspan = 10001

    T = np.zeros((n_tspan,))
    T[nt] = t0
    Y = np.zeros((n_tspan, vsize))
    y0 = DaeIc(dae, y0, t0, opt.rtol)  # check and modify initial values
    yp0 = getyp0(dae, y0, t0)
    Y[0, :] = y0

    events = opt.event
    haveEvent = True if events is not None else False

    if haveEvent:
        value, isterminal, direction = events(t, y0)
        stop = 0
        nevent = -1
        te = np.zeros((10001,))
        ye = np.zeros((10001, vsize))
        ie = np.zeros((10001,))

    Jconstant = False
    Jcurrent = True
    threshold = opt.atol / opt.rtol
    # The initial step size
    if opt.hinit is None:
        if opt.normcontrol:
            pass
        else:
            wt = np.maximum(np.abs(y0), threshold)
            rh = 1.25 * linalg.norm(yp0 / wt, np.inf) / (opt.rtol) ** (1 / 2)
        absh = np.minimum(opt.hmax, tend - t0)
        if absh * rh > 1:
            absh = 1 / rh
        absh = np.maximum(absh, hmin)
    else:
        absh = np.minimum(opt.hmax, np.maximum(hmin, opt.hinit))

    dt = absh

    M = dae.M
    p = dae.p
    done = False

    k = 1
    K = k
    klast = k
    abshlast = absh

    dif = np.zeros((vsize, maxk + 2))
    dif[:, 0] = dt * yp0

    hinvGak = dt * invGa[k - 1]
    nconhk = 0

    if opt.numJac:
        Fty, dFdyt, nfevals = numjac(lambda t_, y_: dae.F(t_, y_, p),
                                     t,
                                     y0,
                                     dae.F(t, y0, p),
                                     1e-5 * np.ones_like(y0),
                                     0,
                                     0,
                                     0,
                                     0)
        dfdy = dFdyt[:, 0:vsize]
    else:
        dfdy = dae.J(t, y0, p)
    stats.nJeval += 1
    Miter = dae.M - hinvGak * dfdy

    rscale = RowScale(Miter)
    Miter = diags_array(rscale, format='csc') @ Miter

    lu = lu_decomposition(Miter)
    stats.ndecomp += 1
    havrate = False

    # THE MAIN LOOP
    done = False
    at_hmin = False
    while not done:
        hmin = 16 * np.spacing(t)
        absh = np.minimum(opt.hmax, np.maximum(hmin, absh))
        if absh == hmin:
            if at_hmin:
                absh = abshlast
            at_hmin = True
        else:
            at_hmin = False
        dt = absh

        # Stretch the step if within 10% of tfinal-t.
        if 1.1 * absh >= abs(tend - t):
            dt = tend - t
            absh = abs(dt)
            done = True

        if (absh != abshlast) or (k != klast):
            # Calculate difRU using cumprod and element-wise operations
            ratio = (kI - 1 - kJ * (absh / abshlast)) / kI
            difRU = np.cumprod(ratio, axis=0) @ difU
            dif[:, 0:K] = dif[:, 0:K] @ difRU[0:K, 0:K]

            hinvGak = dt * invGa[k - 1]
            nconhk = 0
            Miter = dae.M - hinvGak * dfdy

            # Row scaling
            rscale = RowScale(Miter)
            Miter = diags_array(rscale, format='csc') @ Miter

            lu = lu_decomposition(Miter)
            stats.ndecomp = stats.ndecomp + 1
            havrate = False

        # LOOP FOR ADVANCING ONE STEP
        nofailed = True
        while True:
            gotynew = False

            while not gotynew:
                # Compute the constant terms in the equation for y1.
                psi = (dif[:, 0:K] @ (G[0:K].reshape((-1, 1)) * invGa[k - 1])).reshape(-1)

                # Predict a solution at t + dt.
                tnew = t + dt
                if done:
                    tnew = tend  # Hit end point exactly.

                dt = tnew - t  # Purify dt.
                pred = y0 + np.sum(dif[:, 0:K], 1)
                y1 = pred.copy()

                # Initialize difkp1 to zero for the iteration to compute y1.
                difkp1 = np.zeros(vsize)

                if opt.normcontrol:
                    # Compute the norm of y1.
                    normynew = np.linalg.norm(y1)

                    # Calculate invwt and minnrm for norm control.
                    invwt = 1 / max(max(np.linalg.norm(y0), normynew), threshold)
                    minnrm = 100 * np.finfo(float).eps * (normynew * invwt)
                else:
                    # Calculate invwt and minnrm without norm control.
                    abs_y = np.abs(y0)
                    abs_ynew = np.abs(y1)
                    invwt = 1 / np.maximum(np.maximum(abs_y, abs_ynew), threshold)
                    minnrm = 100 * np.finfo(float).eps * np.linalg.norm(y1 * invwt, ord=np.inf)

                tooslow = False
                # Iterate with simplified Newton method
                for iter in range(1, opt.ode15smaxit + 1):

                    # Compute the right-hand side (rhs)
                    rhs = hinvGak * dae.F(tnew, y1, p) - dae.M @ (psi + difkp1)

                    # Account for row scaling
                    rhs = rscale * rhs

                    # Solve the linear system to get del
                    del_ = lu.solve(rhs)  # Assuming odesolve is defined elsewhere

                    # Compute the norm of del
                    if opt.normcontrol:
                        newnrm = np.linalg.norm(del_) * invwt
                    else:
                        newnrm = np.linalg.norm(del_ * invwt, ord=np.inf)

                    # Update difkp1 and y1
                    difkp1 = difkp1 + del_
                    y1 = pred + difkp1

                    # Check convergence criteria
                    if newnrm <= minnrm:
                        gotynew = True
                        break
                    elif iter == 1:
                        if havrate:
                            errit = newnrm * rate / (1 - rate)
                            if errit <= 0.05 * opt.rtol:  # More stringent when using old rate
                                gotynew = True
                                break
                        else:
                            rate = 0
                    elif newnrm > 0.9 * oldnrm:
                        tooslow = True
                        break
                    else:
                        rate = max(0.9 * rate, newnrm / oldnrm)
                        havrate = True
                        errit = newnrm * rate / (1 - rate)
                        if errit <= 0.5 * opt.rtol:
                            gotynew = True
                            break
                        elif iter == opt.ode15smaxit:
                            tooslow = True
                            break
                        elif 0.5 * opt.rtol < errit * rate ** (opt.ode15smaxit - iter):
                            tooslow = True
                            break

                    oldnrm = newnrm
                # Update evaluation and solve counts
                stats.nfeval += iter
                stats.nsolve += iter

                if tooslow:
                    stats.nreject += 1
                    # speed up the iteration by forming new linearization or reducing dt
                    if not Jcurrent:
                        if opt.numJac:
                            Fty, dFdyt, nfevals = numjac(lambda t_, y_: dae.F(t_, y_, p),
                                                         t,
                                                         y0,
                                                         dae.F(t, y0, p),
                                                         1e-5 * np.ones_like(y0),
                                                         0,
                                                         0,
                                                         0,
                                                         0)
                            dfdy = dFdyt[:, 0:vsize]
                        else:
                            dfdy = dae.J(t, y0, p)
                        stats.nJeval += 1
                        Jcurrent = True
                    elif absh <= hmin:
                        print(f"Error exit of RODAS at time = {t}: step size too small dt = {dt}.\n")
                        stats.ret = 'failed'
                        break
                    else:
                        abshlast = absh
                        absh = max(0.3 * absh, hmin)
                        dt = absh
                        done = False

                        ratio = (kI - 1 - kJ * (absh / abshlast)) / kI
                        difRU = np.cumprod(ratio, axis=0) @ difU
                        dif[:, 0:K] = dif[:, 0:K] @ difRU[0:K, 0:K]

                        hinvGak = dt * invGa[k - 1]
                        nconhk = 0
                    Miter = dae.M - hinvGak * dfdy

                    rscale = RowScale(Miter)
                    Miter = diags_array(rscale, format='csc') @ Miter

                    lu = lu_decomposition(Miter)
                    stats.ndecomp += 1
                    havrate = False

            # Calculate the error based on normcontrol flag
            if opt.normcontrol:
                err = np.linalg.norm(difkp1) * invwt * erconst[k - 1]
            else:
                err = np.linalg.norm(difkp1 * invwt, ord=np.inf) * erconst[k - 1]

            # Check for non-negative constraints and update error if necessary
            # if nonNegative and (err <= rtol) and np.any(ynew[idxNonNegative] < 0):
            #     if normcontrol:
            #         errNN = np.linalg.norm(np.maximum(0, -ynew[idxNonNegative])) * invwt
            #     else:
            #         errNN = np.linalg.norm(np.maximum(0, -ynew[idxNonNegative]) / thresholdNonNegative, ord=np.inf)
            #
            #     if errNN > rtol:
            #         err = errNN

            if err > opt.rtol:  # Failed step
                stats.nreject += 1

                if absh <= hmin:
                    print(f"Error exit of RODAS at time = {t}: step size too small dt = {dt}.\n")
                    stats.ret = 'failed'
                    break

                abshlast = absh

                if nofailed:
                    nofailed = False
                    hopt = absh * max(0.1, 0.833 * (opt.rtol / err) ** (1 / (k + 1)))  # 1/1.2

                    if k > 1:
                        if opt.normcontrol:
                            errkm1 = np.linalg.norm(dif[:, k - 1] + difkp1) * invwt * erconst[k - 2]
                        else:
                            errkm1 = np.linalg.norm((dif[:, k - 1] + difkp1) * invwt, ord=np.inf) * erconst[k - 2]

                        hkm1 = absh * max(0.1, 0.769 * (opt.rtol / errkm1) ** (1 / k))  # 1/1.3

                        if hkm1 > hopt:
                            hopt = min(absh, hkm1)  # don't allow step size increase
                            k -= 1
                            K = k

                    absh = max(hmin, hopt)
                else:
                    absh = max(hmin, 0.5 * absh)

                dt = absh

                if absh < abshlast:
                    done = False

                ratio = (kI - 1 - kJ * (absh / abshlast)) / kI
                difRU = np.cumprod(ratio, axis=0) @ difU
                dif[:, 0:K] = dif[:, 0:K] @ difRU[0:K, 0:K]

                hinvGak = dt * invGa[k - 1]
                nconhk = 0
                Miter = dae.M - hinvGak * dfdy

                rscale = RowScale(Miter)
                Miter = diags_array(rscale, format='csc') @ Miter

                lu = lu_decomposition(Miter)
                stats.ndecomp += 1
                havrate = False

            else:  # Successful step
                break

        stats.nstep += 1

        dif[:, k + 1] = difkp1 - dif[:, k]
        dif[:, k] = difkp1
        for j in range(k, 0, -1):
            dif[:, j - 1] += dif[:, j]

        # event detection
        if haveEvent:
            pass

        told = t
        # output sol
        if dense_output:
            while tnew >= tnext > told:
                ynext = ntrp15s(tnext, tnew, y1, dt, dif, k)
                nt = nt + 1
                if nt == n_tspan:
                    n_tspan = n_tspan + 1000
                    T = np.concatenate([T, np.zeros(1000)])
                    Y = np.concatenate([Y, np.zeros((1000, vsize))])
                T[nt] = tnext
                Y[nt] = ynext

                if opt.pbar:
                    pbar.update(T[nt] - T[nt - 1])

                if haveEvent and stop:
                    if tnext >= tevent:
                        break

                inext = inext + 1
                if inext <= n_tspan - 1:
                    tnext = tspan[inext]
                    if haveEvent and stop:
                        if tnext >= tevent:
                            tnext = tevent
                else:
                    tnext = tend + dt

        else:
            nt += 1
            if nt == n_tspan:
                n_tspan = n_tspan + 1000
                T = np.concatenate([T, np.zeros(1000)])
                Y = np.concatenate([Y, np.zeros((1000, vsize))])
            T[nt] = tnew
            Y[nt] = y1

            if opt.pbar:
                pbar.update(T[nt] - T[nt - 1])

        if done:
            break

        klast = k
        abshlast = absh

        nconhk = min(nconhk + 1, maxk + 2)

        if nconhk >= k + 2:
            temp = 1.2 * (err / opt.rtol) ** (1 / (k + 1))
            if temp > 0.1:
                hopt = absh / temp
            else:
                hopt = 10 * absh
            kopt = k

            # reduce order?
            if k > 1:
                if opt.normcontrol:
                    errkm1 = np.linalg.norm(dif[:, k - 1]) * invwt * erconst[k - 2]
                else:
                    errkm1 = np.linalg.norm(dif[:, k - 1] * invwt, ord=np.inf) * erconst[k - 2]

                temp = 1.3 * (errkm1 / opt.rtol) ** (1 / k)
                if temp > 0.1:
                    hkm1 = absh / temp
                else:
                    hkm1 = 10 * absh

                if hkm1 > hopt:
                    hopt = hkm1
                    kopt = k - 1

            # increase order?
            if k < maxk:
                if opt.normcontrol:
                    errkp1 = np.linalg.norm(dif[:, k + 1]) * invwt * erconst[k]
                else:
                    errkp1 = np.linalg.norm(dif[:, k + 1] * invwt, ord=np.inf) * erconst[k]

                temp = 1.4 * (errkp1 / opt.rtol) ** (1 / (k + 2))
                if temp > 0.1:
                    hkp1 = absh / temp
                else:
                    hkp1 = 10 * absh

                if hkp1 > hopt:
                    hopt = hkp1
                    kopt = k + 1

            # update dt and order
            if hopt > absh:
                absh = hopt
                if k != kopt:
                    k = kopt
                    K = k

        # update the integration one step
        t = tnew
        y0 = y1

        if opt.normcontrol:
            normy = normynew
        Jcurrent = Jconstant

    T = T[0:nt + 1]
    Y = Y[0:nt + 1]

    if opt.pbar:
        pbar.close()

    if haveEvent:
        pass
        # te = te[0:nevent + 1]
        # ye = ye[0:nevent + 1]
        # ie = ie[0:nevent + 1]
        # return daesol(T, Y, te, ye, ie, stats)
    else:
        return daesol(T, Y, stats=stats)


def RowScale(Miter):
    # TODO: RowScale performs the row manipulation, which is not suitable for csc type

    RowScale = 1 / np.max(np.abs(Miter), axis=1).toarray()

    return RowScale
