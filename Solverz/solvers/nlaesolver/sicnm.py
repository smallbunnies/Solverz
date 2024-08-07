import warnings

from Solverz.solvers.nlaesolver.utilities import *
from Solverz.solvers.daesolver.rodas.param import Rodas_param
from scipy.sparse import eye_array as speye
from scipy.sparse.linalg import splu, spsolve_triangular
from scipy.sparse import csc_array, block_array
from Solverz.solvers.parser import dae_io_parser
from Solverz.solvers.solution import daesol


@ae_io_parser
def sicnm(ae: nAE,
          y0: np.ndarray,
          opt: Opt = None):
    r"""
    The semi-implicit continuous Newton method [1]_. The philosophy is to rewrite the algebraic equation


        .. math::

            0=g(y)
        
    as the differential algebraic equations

        .. math::

            \begin{aligned}
                \dot{y}&=z \\
                0&=J(y)z+g(y)
            \end{aligned}

    with $y_0$ being the initial value guess and $z_0=-J(y_0)^{-1}g(y_0)$, where $z$ is an intermediate variable introduced. Then the DAEs are solved by Rodas. SICNM is found to be more robust than the Newton's method, for which the theoretical proof can be found in my paper [1]_. In addition, the non-iterative nature of Rodas guarantees the efficiency.

    One can change the rodas scheme according to the ones implemented in the DAE version of Rodas.
    
    SICNM used the Hessian-vector product (HVP) interface of the algebraic equation models, the `make_hvp` flag should be set to `True` when printing numerical modules. 

    An illustrative example of SICNM can be found in the `power flow section <https://cook.solverz.org/ae/pf/pf.html>`_ of Solverz' cookbook.

    Parameters
    ==========

    ae : nAE
        Numerical AE object.

    y0 : np.ndarray
        The initial values of variables

    opt : Opt
        The solver options, including:

        - ite_tol: 1e-5(default)|float
            The iteration error tolerance.

    Returns
    =======

    sol : aesol
        The aesol object.

    References
    ==========

    .. [1] R. Yu, W. Gu, S. Lu, and Y. Xu, “Semi-implicit continuous newton method for power flow analysis,” 2023, arXiv:2312.02809.


    """
    p = ae.p

    def F_tilda(y_, v_):
        return np.concatenate([v_, ae.J(y_, p) @ v_ + ae.F(y_, p)])

    if opt is None:
        opt = Opt()
    stats = Stats('Sicnm based on ' + opt.scheme)

    rparam = Rodas_param(opt.scheme)
    vsize = y0.shape[0]
    tspan = np.array([0, 10000])
    tend = tspan[-1]
    t0 = tspan[0]
    if opt.hmax is None:
        opt.hmax = np.abs(tend - t0)
    nt = 0
    t = t0
    hmin = 16 * np.spacing(t0)
    uround = np.spacing(1.0)
    T = np.zeros((10001,))
    T[nt] = t0
    y = np.zeros((10001, vsize))
    v = np.zeros((10001, vsize))
    y[0, :] = y0
    v0 = solve(ae.J(y0, p), -ae.F(y0, p))
    v[0, :] = v0

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

    ZERO = None
    EYE = speye(vsize, format='csc')
    M = csc_array((np.ones(vsize), (np.arange(0, vsize), np.arange(0, vsize))),
                  shape=(2 * vsize, 2 * vsize))
    done = False
    reject = 0
    while not done:
        # step size too small
        # pass

        if np.abs(dt) < uround:
            print(f"Error exit of RODAS at time = {t}: step size too small h = {dt}.\n")
            stats.ret = 'failed'
            stats.succeed = False
            break

        if reject > 100:
            print(f"Step rejected over 100 times at time = {t}.\n")
            stats.ret = 'failed'
            stats.succeed = False
            break

        # Stretch the step if within 10% of T-t.
        if t + dt >= tend:
            dt = tend - t
        else:
            dt = np.minimum(dt, 0.5 * (tend - t))

        if done:
            break
        K = np.zeros((2 * vsize, rparam.s))

        if reject == 0:
            Hvp = ae.HVP(y0, p, v0)
            J = ae.J(y0, p)

        dfdt0 = 0
        rhs = F_tilda(y0, v0) + rparam.g[0] * dfdt0
        stats.nfeval = stats.nfeval + 1

        try:
            if opt.partial_decompose:
                # partial decomposition
                N = - dt * rparam.gamma * (Hvp + J)
                Lambda = dt * rparam.gamma * (N - J)
                lu = splu(Lambda)
                if nt == 0:
                    P = csc_array((np.ones(vsize), (lu.perm_r, np.arange(vsize))))
                    Q = csc_array((np.ones(vsize), (np.arange(vsize), lu.perm_c)))
                    # b_perm = np.concatenate([np.arange(vsize), lu.perm_r + vsize])
                    # dx_perm = np.concatenate([np.arange(vsize), lu.perm_c + vsize])
                    P_tilda = block_array([[EYE, ZERO], [ZERO, P]], format='csc')
                    Q_tilda = block_array([[EYE, ZERO], [ZERO, Q]], format='csc')

                    L_tilda = block_array([[EYE, ZERO], [P @ N, lu.L]], format='csc')
                    U_tilda = block_array([[EYE, -dt * rparam.gamma * Q], [ZERO, lu.U]], format='csc')
            else:
                # full decomposition
                tilde_J = block_array([[ZERO, EYE], [Hvp + J, J]])
                tilde_E = M - dt * rparam.gamma * tilde_J
                lu = splu(tilde_E)
        except RuntimeError:
            stats.succeed = False
            break

        stats.ndecomp = stats.ndecomp + 1
        if opt.partial_decompose:
            # partial decomposition
            # K[dx_perm, 0] = solve(U_tilda, solve(L_tilda, rhs[b_perm]))
            K[:, 0] = Q_tilda@(solve(U_tilda, solve(L_tilda, P_tilda@rhs)))
            # not stable be very careful with the following triangular solver
            # K[:, 0] = Q_tilda@(spsolve_triangular(U_tilda, spsolve_triangular(L_tilda, P_tilda@rhs), False))
        else:
            # full decomposition
            K[:, 0] = lu.solve(rhs)
            # K[:, 0] = solve(tilde_E, rhs)

        for j in range(1, rparam.s):
            sum_1 = K @ rparam.alpha[:, j]
            sum_2 = K @ rparam.gammatilde[:, j]
            y1 = y0 + dt * sum_1[0:vsize]
            v1 = v0 + dt * sum_1[vsize:2 * vsize]

            rhs = F_tilda(y1, v1) + M @ sum_2 + rparam.g[j] * dfdt0
            stats.nfeval = stats.nfeval + 1

            if opt.partial_decompose:
                # partial decomposition
                # K[dx_perm, j] = solve(U_tilda, solve(L_tilda, rhs[b_perm]))
                K[:, j] = Q_tilda @ (solve(U_tilda, solve(L_tilda, P_tilda @ rhs)))
                # not stable be very careful with the following triangular solver
                # K[:, j] = Q_tilda @ (spsolve_triangular(U_tilda, spsolve_triangular(L_tilda, P_tilda @ rhs), False))
            else:
                # full decomposition
                K[:, j] = lu.solve(rhs)
                # K[:, j] = solve(tilde_E, rhs)

            K[:, j] = K[:, j] - sum_2

        sum_1 = K @ (dt * rparam.b)
        ynew = y0 + sum_1[0:vsize]
        vnew = v0 + sum_1[vsize:2 * vsize]
        sum_2 = K @ (dt * rparam.bd)
        y_tilda = np.concatenate([ynew, vnew])
        SK = (opt.atol + opt.rtol * abs(y_tilda)).reshape((-1,))
        err = np.max(np.abs((sum_1 - sum_2) / SK))
        if np.any(np.isinf(y_tilda)) or np.any(np.isnan(y_tilda)):
            err = 1.0e6
            print('Warning Rodas: NaN or Inf occurs.')
        err = np.maximum(err, 1.0e-6)
        fac = opt.f_savety / (err ** (1 / rparam.pord))
        fac = np.minimum(opt.facmax, np.maximum(opt.fac1, fac))
        dtnew = dt * fac

        if err <= 1.0:
            reject = 0
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
                        v0_ = valueold[i]
                        v1_ = value[i]
                        detect = 1
                        if direction[i] < 0 and v0_ <= v1_:
                            detect = 0
                        if direction[i] > 0 and v0_ >= v1_:
                            detect = 0
                        if detect:
                            iterate = 1
                            tL = told
                            tR = t
                            if np.abs(v1_ - v0_) > uround:
                                tevent = told - v0_ * dt / (v1_ - v0_)  # initial guess for tevent
                            else:
                                iterate = 0
                                tevent = t
                                ynext = ynew

                            tol = 128 * np.max([np.spacing(told), np.spacing(t)])
                            tol = np.min([tol, np.abs(t - told)])
                            while iterate > 0:
                                iterate = iterate + 1
                                tau = (tevent - told) / dt
                                ynext = y0 + (tau * dt * K @ (rparam.b + (tau - 1) * (rparam.c + tau * (rparam.d + tau * rparam.e))))[0:vsize]
                                value, isterminal, direction = events(tevent, ynext)
                                if v1_ * value[i] < 0:
                                    tL = tevent
                                    tevent = 0.5 * (tevent + tR)
                                    v0_ = value[i]
                                elif v0_ * value[i] < 0:
                                    tR = tevent
                                    tevent = 0.5 * (tL + tevent)
                                    v1_ = value[i]
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
                    y[nt] = ynext

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
                y[nt] = ynew
                v[nt] = vnew

            if nt == 10000:
                warnings.warn("Time steps more than 10000! Rodas breaks. Try input a smaller tspan!")
                stats.succeed = False
                done = True

            if np.abs(tend - t) < uround:
                done = True
            if np.max(np.abs(ae.F(ynew, p))) < opt.ite_tol:
                done = True
            if stop:
                done = True

            y0 = ynew
            v0 = vnew
            opt.facmax = opt.fac2

        else:
            reject = reject + 1
            stats.nreject = stats.nreject + 1
            opt.facmax = 1
        dt = np.min([opt.hmax, np.max([hmin, dtnew])])

    T = T[0:nt + 1]
    y = y[0:nt + 1]

    sol = aesol(y[-1], stats=stats)

    stats.T = T
    stats.y = y
    if haveEvent:
        stats.te = te[0:nevent + 1]
        stats.ye = ye[0:nevent + 1]
        stats.ie = ie[0:nevent + 1]

    return sol
