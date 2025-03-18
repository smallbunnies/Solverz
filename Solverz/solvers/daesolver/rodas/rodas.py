import warnings

from Solverz.solvers.daesolver.utilities import *
from Solverz.solvers.daesolver.rodas.param import Rodas_param


@dae_io_parser
def Rodas(dae: nDAE,
          tspan: List | np.ndarray,
          y0: np.ndarray,
          opt: Opt = None):
    r"""
    The stiffly accurate Rosenbrock methods including Rodas4 [1]_, Rodasp [2]_, Rodas5p [3]_.

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

        - scheme: 'rodas4'(default)|'rodasp'|'rodas5p'
            The rodas scheme
        - rtol: 1e-3(default)|float
            The relative error tolerance
        - atol: 1e-6(default)|float
            The absolute error tolerance
        - event: Callable
            The simulation events, with $t$ and $y$ being args,
            and `value`, `is_terminal` and `direction` being outputs
        - fixed_h: False(default)|bool
            To use fixed step size
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

    .. [1] Hairer and Wanner, Solving Ordinary Differential Equations II , 2nd ed. Berlin, Heidelberg, Germany: Springer-Verlag, 1996.
    .. [2] Steinebach, Order-reduction of ROW-methods for DAEs and method of lines applications. Preprint-Nr. 1741, FB Mathematik, TH Darmstadt, 1995
    .. [3] Steinebach, “Construction of rosenbrock-wanner method rodas5p and numerical benchmarks within the julia differential equations package,” BIT, vol. 63, no. 27, Jun 2023.
    """

    if opt is None:
        opt = Opt()
    stats = Stats(opt.scheme)

    rparam = Rodas_param(opt.scheme)
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
    T = np.zeros((10001,))
    T[nt] = t0
    Y = np.zeros((10001, vsize))
    y0 = DaeIc(dae, y0, t0, opt.rtol)  # check and modify initial values
    Y[0, :] = y0

    if opt.pbar:
        pbar = tqdm(total=tend - t0)

    dense_output = False
    n_tspan = len(tspan)
    told = t0
    if n_tspan > 2:
        dense_output = True
        inext = 1
        tnext = tspan[inext]

    events = opt.event
    haveEvent = True if events is not None else False

    if haveEvent:
        value, isterminal, direction = events(t, y0)
    stop = 0
    nevent = -1
    te = np.zeros((10001,))
    ye = np.zeros((10001, vsize))
    ie = np.zeros((10001,))

    # The initial step size
    if opt.hinit is None:
        dt = 1e-6 * (tend - t0)
    else:
        dt = opt.hinit

    dt = np.maximum(dt, hmin)
    dt = np.minimum(dt, opt.hmax)

    M = dae.M
    p = dae.p
    done = False
    reject = 0
    while not done:
        # step size too small
        # pass

        if np.abs(dt) < uround:
            print(f"Error exit of RODAS at time = {t}: step size too small h = {dt}.\n")
            stats.ret = 'failed'
            break

        if reject > 100:
            print(f"Step rejected over 100 times at time = {t}.\n")
            stats.ret = 'failed'
            break

        # Stretch the step if within 10% of T-t.
        if t + dt >= tend:
            dt = tend - t
        else:
            dt = np.minimum(dt, 0.5 * (tend - t))

        if opt.fix_h:
            dt = opt.hinit

        if done:
            break
        K = np.zeros((vsize, rparam.s))

        if reject == 0:
            J = dae.J(t, y0, p)
            stats.nJeval += 1

        dfdt0 = dt * dfdt(dae, t, y0)
        rhs = dae.F(t, y0, p) + rparam.g[0] * dfdt0
        stats.nfeval = stats.nfeval + 1

        try:
            lu = lu_decomposition(M - dt * rparam.gamma * J)
        except RuntimeError:
            break
        stats.ndecomp = stats.ndecomp + 1
        K[:, 0] = lu.solve(rhs)

        for j in range(1, rparam.s):
            sum_1 = K @ rparam.alpha[:, j]
            sum_2 = K @ rparam.gammatilde[:, j]
            y1 = y0 + dt * sum_1

            rhs = dae.F(t + dt * rparam.a[j], y1, p) + M @ sum_2 + rparam.g[j] * dfdt0
            stats.nfeval = stats.nfeval + 1
            sol = lu.solve(rhs)
            K[:, j] = sol - sum_2
        stats.nsolve += rparam.s

        sum_1 = K @ (dt * rparam.b)
        ynew = y0 + sum_1
        if not opt.fix_h:
            sum_2 = K @ (dt * rparam.bd)
            SK = (opt.atol + opt.rtol * abs(ynew)).reshape((-1,))
            err = np.max(np.abs((sum_1 - sum_2) / SK))
            if np.any(np.isinf(ynew)) or np.any(np.isnan(ynew)):
                err = 1.0e6
                print('Warning Rodas: NaN or Inf occurs.')
            err = np.maximum(err, 1.0e-6)
            fac = opt.f_savety / (err ** (1 / rparam.pord))
            fac = np.minimum(opt.facmax, np.maximum(opt.fac1, fac))
            dtnew = dt * fac
        else:
            err = 1.0
            dtnew = dt

        if err <= 1.0:
            reject = 0
            told = t
            t = t + dt

            stats.nstep = stats.nstep + 1
            # events
            if haveEvent:
                valueold = value
                value, isterminal, direction = events(t, ynew)
                value_save = value
                ff = np.where(value * valueold < 0)[0]
                if ff.size > 0:
                    for i in ff:
                        v0 = valueold[i]
                        v1 = value[i]
                        detect = 1
                        if direction[i] < 0 and v0 <= v1:
                            detect = 0
                        if direction[i] > 0 and v0 >= v1:
                            detect = 0
                        if detect:
                            iterate = 1
                            tL = told
                            tR = t
                            if np.abs(v1 - v0) > uround:
                                tevent = told - v0 * dt / (v1 - v0)  # initial guess for tevent
                            else:
                                iterate = 0
                                tevent = t
                                ynext = ynew

                            tol = 128 * np.max([np.spacing(told), np.spacing(t)])
                            tol = np.min([tol, np.abs(t - told)])
                            while iterate > 0:
                                iterate = iterate + 1
                                tau = (tevent - told) / dt
                                ynext = y0 + tau * dt * K @ (
                                        rparam.b + (tau - 1) * (rparam.c + tau * (rparam.d + tau * rparam.e)))
                                value, isterminal, direction = events(tevent, ynext)
                                if v1 * value[i] < 0:
                                    tL = tevent
                                    tevent = 0.5 * (tevent + tR)
                                    v0 = value[i]
                                elif v0 * value[i] < 0:
                                    tR = tevent
                                    tevent = 0.5 * (tL + tevent)
                                    v1 = value[i]
                                else:
                                    iterate = 0
                                if (tR - tL) < tol:
                                    iterate = 0
                                if iterate > 100:
                                    print(f"Lost Event in interval [{told}, {t}].\n")
                                    break
                            if np.abs(tevent - told) < opt.event_duration:
                                # We're not going to find events closer than tol.
                                break
                            t = tevent
                            ynew = ynext
                            nevent += 1
                            te[nevent] = tevent
                            ie[nevent] = i
                            ye[nevent] = ynext
                            value, isterminal, direction = events(tevent, ynext)
                            value = value_save
                            if isterminal[i]:
                                if dense_output:
                                    if tnext >= tevent:
                                        tnext = tevent
                                stop = 1
                                break

            if dense_output:  # dense_output
                while t >= tnext > told:
                    tau = (tnext - told) / dt
                    ynext = y0 + tau * dt * K @ (rparam.b + (tau - 1) * (rparam.c + tau * (rparam.d + tau * rparam.e)))
                    nt = nt + 1
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
                nt = nt + 1
                T[nt] = t
                Y[nt] = ynew

                if opt.pbar:
                    pbar.update(T[nt] - T[nt - 1])

            if nt == 10000:
                warnings.warn("Time steps more than 10000! Rodas breaks. Try input a smaller tspan!")
                done = True

            if np.abs(tend - t) < uround or stop:
                done = True
            y0 = ynew
            opt.facmax = opt.fac2

        else:
            reject = reject + 1
            stats.nreject = stats.nreject + 1
            opt.facmax = 1
        dt = np.min([opt.hmax, np.max([hmin, dtnew])])

    T = T[0:nt + 1]
    Y = Y[0:nt + 1]

    if opt.pbar:
        pbar.close()

    if haveEvent:
        te = te[0:nevent + 1]
        ye = ye[0:nevent + 1]
        ie = ie[0:nevent + 1]
        return daesol(T, Y, te, ye, ie, stats)
    else:
        return daesol(T, Y, stats=stats)


def dfdt(dae: nDAE, t, y):
    tscale = np.maximum(0.1 * np.abs(t), 1e-8)
    ddt = t + np.sqrt(np.spacing(1)) * tscale - t
    f0 = dae.F(t, y, dae.p)
    f1 = dae.F(t + ddt, y, dae.p)
    return (f1 - f0) / ddt
