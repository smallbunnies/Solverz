import warnings

from Solverz.solvers.daesolver.utilities import *


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
    if opt.hmax is None:
        opt.hmax = np.abs(tend - t0)
    nt = 0
    t = t0
    hmin = 16 * np.spacing(t0)
    uround = np.spacing(1.0)
    T = np.zeros((10001,))
    T[nt] = t0
    y = np.zeros((10001, vsize))
    y0 = DaeIc(dae, y0, t0, opt.rtol)  # check and modify initial values
    y[0, :] = y0

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
    y = y[0:nt + 1]
    if haveEvent:
        te = te[0:nevent + 1]
        ye = ye[0:nevent + 1]
        ie = ie[0:nevent + 1]
        return daesol(T, y, te, ye, ie, stats)
    else:
        return daesol(T, y, stats=stats)


class Rodas_param:

    def __init__(self,
                 scheme: str = 'rodas4'):
        match scheme:
            case 'rodas4':
                self.s = 6
                self.pord = 4
                self.alpha = np.zeros((self.s, self.s))
                self.beta = np.zeros((self.s, self.s))
                self.g = np.zeros((self.s, 1))
                self.gamma = 0.25
                self.alpha[1, 0] = 3.860000000000000e-01
                self.alpha[2, 0:2] = [1.460747075254185e-01, 6.392529247458190e-02]
                self.alpha[3, 0:3] = [-3.308115036677222e-01, 7.111510251682822e-01, 2.496604784994390e-01]
                self.alpha[4, 0:4] = [-4.552557186318003e+00, 1.710181363241323e+00, 4.014347332103149e+00,
                                      -1.719715090264703e-01]
                self.alpha[5, 0:5] = [2.428633765466977e+00, -3.827487337647808e-01, -1.855720330929572e+00,
                                      5.598352992273752e-01,
                                      2.499999999999995e-01]
                self.beta[1, 0] = 3.170000000000250e-02
                self.beta[2, 0:2] = [1.247220225724355e-02, 5.102779774275723e-02]
                self.beta[3, 0:3] = [1.196037669338736e+00, 1.774947364178279e-01, -1.029732405756564e+00]
                self.beta[4, 0:4] = [2.428633765466977e+00, -3.827487337647810e-01, -1.855720330929572e+00,
                                     5.598352992273752e-01]
                self.beta[5, 0:5] = [3.484442712860512e-01, 2.130136219118989e-01, -1.541025326623184e-01,
                                     4.713207793914960e-01,
                                     -1.286761399271284e-01]
                self.b = np.zeros((6,))
                self.b[0:5] = self.beta[5, 0:5]
                self.b[5] = self.gamma
                self.bd = np.zeros((6,))
                self.bd[0:4] = self.beta[4, 0:4]
                self.bd[4] = self.gamma
                self.c = np.array([-4.786970949443344e+00, -6.966969867338157e-01, 4.491962205414260e+00,
                                   1.247990161586704e+00, -2.562844308238056e-01, 0])
                self.d = np.array([1.274202171603216e+01, -1.894421984691950e+00, -1.113020959269748e+01,
                                   -1.365987420071593e+00, 1.648597281428871e+00, 0])
                self.e = np.zeros((6,))
                self.gammatilde = self.beta - self.alpha
                self.a = np.sum(self.alpha, axis=1)
                self.g = np.sum(self.gammatilde, axis=1) + self.gamma
                self.gammatilde = self.gammatilde / self.gamma
                self.alpha = self.alpha.T
                self.gammatilde = self.gammatilde.T
            case 'rodasp':
                self.s = 6
                self.pord = 4
                self.alpha = np.zeros((self.s, self.s))
                self.beta = np.zeros((self.s, self.s))
                self.gamma = 0.25
                self.alpha[1, 0] = 0.75
                self.alpha[2, 0:2] = [0.0861204008141522, 0.123879599185848]
                self.alpha[3, 0:3] = [0.774934535507324, 0.149265154950868, -0.294199690458192]
                self.alpha[4, 0:4] = [5.30874668264614, 1.33089214003727, -5.37413781165556, -0.265501011027850]
                self.alpha[5, 0:5] = [-1.76443764877448, -0.474756557206303, 2.36969184691580, 0.619502359064983,
                                      0.250000000000000]
                self.beta[1, 0] = 0.0
                self.beta[2, 0:2] = [-0.0493920000000000, -0.0141120000000000]
                self.beta[3, 0:3] = [-0.482049469387756, -0.100879555555556, 0.926729024943312]
                self.beta[4, 0:4] = [-1.76443764877448, -0.474756557206303, 2.36969184691580, 0.619502359064983]
                self.beta[5, 0:5] = [-0.0803683707891135, -0.0564906135924476, 0.488285630042799, 0.505716211481619,
                                     -0.107142857142857]
                self.b = np.zeros((6,))
                self.b[0:5] = self.beta[5, 0:5]
                self.b[5] = self.gamma
                self.bd = np.zeros((6,))
                self.bd[0:4] = self.beta[4, 0:4]
                self.bd[4] = self.gamma
                self.c = np.array([-40.98639964388325,
                                   -10.36668980524365,
                                   44.66751816647147,
                                   4.13001572709988,
                                   2.55555555555556,
                                   0])
                self.d = np.array([73.75018659483291,
                                   18.54063799119389,
                                   -81.66902074619779,
                                   -6.84402606205123,
                                   -3.77777777777778,
                                   0])
                self.e = np.zeros((6,))
                self.gammatilde = self.beta - self.alpha
                self.a = np.sum(self.alpha, axis=1)
                self.g = np.sum(self.gammatilde, axis=1) + self.gamma
                self.gammatilde = self.gammatilde / self.gamma
                self.alpha = self.alpha.T
                self.gammatilde = self.gammatilde.T
            case 'rodas5p':
                self.s = 8
                self.pord = 5
                self.a = np.zeros((self.s,))
                self.g = np.zeros((self.s,))
                self.alpha = np.array([
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.6358126895828704, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.31242290829798824, 0.09715693104176527, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.3140825753299277, 1.8583084874257945, -2.1954603902496506, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.42153145792835994, 0.25386966273009, -0.2365547905326239, -0.010005969169959593, 0.0, 0.0, 0.0,
                     0.0],
                    [1.712028062121536, 2.4456320333807953, -3.117254839827603, -0.04680538266310614,
                     0.006400126988377645, 0.0, 0.0, 0.0],
                    [-0.9993030215739269, -1.5559156221686088, 3.1251564324842267, 0.24141811637172583,
                     -0.023293468307707062, 0.21193756319429014, 0.0, 0.0],
                    [-0.003487250199264519, -0.1299669712056423, 1.525941760806273, 1.1496140949123888,
                     -0.7043357115882416, -1.0497034859198033, 0.21193756319429014, 0.0]
                ])
                self.beta = np.array([
                    [0.21193756319429014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.21193756319429014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.10952700614965587, -0.03129343032847311, 0.21193756319429014, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.701745865188331, -0.15675801606110462, 1.024650547472829, 0.21193756319429014, 0.0, 0.0, 0.0,
                     0.0],
                    [3.587261990937329, 1.6112735397639253, -2.378003402448709, -0.2778036907258995,
                     0.21193756319429014, 0.0, 0.0, 0.0],
                    [-0.9993030215739269, -1.5559156221686088, 3.1251564324842267, 0.24141811637172583,
                     -0.023293468307707062, 0.21193756319429014, 0.0, 0.0],
                    [-0.003487250199264519, -0.1299669712056423, 1.525941760806273, 1.1496140949123888,
                     -0.7043357115882416, -1.0497034859198033, 0.21193756319429014, 0.0],
                    [0.12236007991300712, 0.050238881884191906, 1.3238392208663585, 1.2643883758622305,
                     -0.7904031855871826, -0.9680932754194287, -0.214267660713467, 0.21193756319429014]
                ])
                self.c = np.array([-0.8232744916805133, 0.3181483349120214, 0.16922330104086836, -0.049879453396320994,
                                   0.19831791977261218, 0.31488148287699225, -0.16387506167704194,
                                   0.036457968151382296])
                self.d = np.array([-0.6726085201965635, -1.3128972079520966, 9.467244336394248, 12.924520918142036,
                                   -9.002714541842755, -11.404611057341922, -1.4210850083209667,
                                   1.4221510811179898])
                self.e = np.array([1.4025185206933914, 0.9860299407499886, -11.006871867857507, -14.112585514422294,
                                   9.574969612795117, 12.076626078349426, 2.114222828697341,
                                   -1.0349095990054304])
                self.gamma = self.beta[0, 0]
                self.b = np.append(self.beta[7, :-1], [self.gamma])
                self.bd = np.append(self.beta[6, :-2], [self.gamma, 0])
                self.gammatilde = self.beta - self.gamma * np.eye(self.s) - self.alpha
                for i in range(self.s):
                    self.a[i] = np.sum(self.alpha[i, :])
                    self.g[i] = np.sum(self.gammatilde[i, :]) + self.gamma
                self.gammatilde /= self.gamma
                self.alpha = self.alpha.T
                self.gammatilde = self.gammatilde.T
            case 'rodas3d':
                self.s = 4
                self.pord = 3
                self.alpha = np.zeros((self.s, self.s))
                self.beta = np.zeros((self.s, self.s))
                self.g = np.zeros((self.s, 1))
                self.gamma = 0.57281606
                self.alpha[1, 0] = 1.2451051999132263
                self.alpha[2, 0:2] = [1, 0]
                self.alpha[3, 0:3] = [0.32630307266483527, 0.10088086733516474, 0.57281606]
                self.beta[1, 0] = -3.1474142698552949
                self.beta[2, 0:2] = [0.32630307266483527, 0.10088086733516474]
                self.beta[3, 0:3] = [0.69775271462407906, 0.056490613592447572, -0.32705938821652658]
                self.b = np.zeros((self.s,))
                self.b[0:3] = self.beta[3, 0:3]
                self.b[self.s - 1] = self.gamma
                self.bd = np.zeros((self.s,))
                self.bd[0:2] = self.beta[2, 0:2]
                self.bd[self.s - 2] = self.gamma
                # self.c = np.array([-4.786970949443344e+00, -6.966969867338157e-01, 4.491962205414260e+00,
                #                    1.247990161586704e+00, -2.562844308238056e-01, 0])
                # self.d = np.array([1.274202171603216e+01, -1.894421984691950e+00, -1.113020959269748e+01,
                #                    -1.365987420071593e+00, 1.648597281428871e+00, 0])
                # self.e = np.zeros((6,))
                self.gammatilde = self.beta - self.alpha
                self.a = np.sum(self.alpha, axis=1)
                self.g = np.sum(self.gammatilde, axis=1) + self.gamma
                self.gammatilde = self.gammatilde / self.gamma
                self.alpha = self.alpha.T
                self.gammatilde = self.gammatilde.T
            case _:
                raise ValueError("Not implemented")


def dfdt(dae: nDAE, t, y):
    tscale = np.maximum(0.1 * np.abs(t), 1e-8)
    ddt = t + np.sqrt(np.spacing(1)) * tscale - t
    f0 = dae.F(t, y, dae.p)
    f1 = dae.F(t + ddt, y, dae.p)
    return (f1 - f0) / ddt
