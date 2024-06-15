import warnings

from Solverz.solvers.nlaesolver.utilities import *
from Solverz.solvers.daesolver.rodas import Rodas_param
from scipy.sparse import eye_array as speye
from scipy.sparse.linalg import splu
from scipy.sparse import csc_array, block_array


@ae_io_parser
def sicnm(ae: nAE,
          y0: np.ndarray,
          opt: Opt = None):

    p = ae.p

    def F_tilda(y_, v_):
        return np.concatenate([v_, ae.J(y_, p) @ v_ + ae.F(y_, p)])

    if opt is None:
        opt = Opt()
    stats = Stats(opt.scheme)

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

        if done:
            break
        K = np.zeros((vsize, rparam.s))

        if reject == 0:
            Hvp = ae.Hvp(y0, p, v0)
            J = ae.J(y0, p)

        dfdt0 = 0
        rhs = F_tilda(y0, v0) + rparam.g[0] * dfdt0
        stats.nfeval = stats.nfeval + 1

        try:
            N = - dt * rparam.gamma * (Hvp + J)
            Lambda = dt * rparam.gamma * (N - J)
            lu = splu(Lambda)
            if nt == 0:
                P = csc_array((np.ones(vsize), (lu.perm_r, np.arange(vsize))))
                Q = csc_array((np.ones(vsize), (np.arange(vsize), lu.perm_c)))
                b_perm = np.concatenate([np.arange(vsize), lu.perm_r + vsize])
                dx_perm = np.concatenate([np.arange(vsize), lu.perm_c + vsize])

            L_tilda = block_array([[EYE, ZERO], [P @ N, lu.L]])
            U_tilda = block_array([[EYE, -dt * rparam.gamma * Q], [ZERO, lu.U]])
        except RuntimeError:
            break

        stats.ndecomp = stats.ndecomp + 1
        K[dx_perm, 0] = solve(U_tilda, solve(L_tilda, rhs[b_perm]))

        for j in range(1, rparam.s):
            sum_1 = K @ rparam.alpha[:, j]
            sum_2 = K @ rparam.gammatilde[:, j]
            y1 = y0 + dt * sum_1[0:vsize]
            v1 = v0 + dt * sum_1[vsize:2 * vsize]

            rhs = F_tilda(y1, v1) + M @ sum_2 + rparam.g[j] * dfdt0
            stats.nfeval = stats.nfeval + 1
            K[dx_perm, j] = solve(U_tilda, solve(L_tilda, rhs[b_perm]))
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
            nt = nt + 1
            T[nt] = t
            y[nt] = ynew
            v[nt] = vnew

            if nt == 10000:
                warnings.warn("Time steps more than 10000! Rodas breaks. Try input a smaller tspan!")
                done = True

            if np.abs(tend - t) < uround:
                done = True
            if np.max(np.abs(ae.F(ynew, p)) < opt.ite_tol):
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
    return aesol(y[-1], stats=stats)
