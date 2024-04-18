from sympy import simplify

from Solverz.sym_algebra.symbols import Para, iAliasVar, IdxVar, iVar
from Solverz.sym_algebra.functions import switch
from Solverz.utilities.type_checker import is_number


def finite_difference(diff_var, flux, source, two_dim_var, M, scheme='central diff', direction=None, dx=None):
    M_ = M + 1  # for pretty printer of M in slice
    if dx is None:
        dx = Para('dx')
    else:
        if is_number(dx):
            dx = dx
        else:
            raise TypeError(f'Input dx is not number!')

    dt = Para('dt')
    u = diff_var
    u0 = iAliasVar(u.name + '_tag_0')

    if scheme == 'central diff':
        fui1j1 = flux.subs([(a, a[1:M_]) for a in two_dim_var])
        fuij1 = flux.subs([(a, a[0:M_ - 1]) for a in two_dim_var])
        fui1j = flux.subs([(a, iAliasVar(a.name + '_tag_0')[1:M_]) for a in two_dim_var])
        fuij = flux.subs([(a, iAliasVar(a.name + '_tag_0')[0:M_ - 1]) for a in two_dim_var])

        S = source.subs([(a, (a[1:M_] + a[0:M_ - 1] + iAliasVar(a.name + '_tag_0')[1:M_] + iAliasVar(a.name + '_tag_0')[
                                                                                          0:M_ - 1]) / 4) for a in
                         two_dim_var])

        fde = dx * (u[1:M_] - u0[1:M_] + u[0:M_ - 1] - u0[0:M_ - 1]) \
              + simplify(dt * (fui1j1 - fuij1 + fui1j - fuij)) \
              - simplify(2 * dx * dt * S)

    elif scheme == 'euler':
        if direction == 1:
            fui1j1 = flux.subs([(a, a[1:M_]) for a in two_dim_var])
            fuij1 = flux.subs([(a, a[0:M_ - 1]) for a in two_dim_var])
            S = source.subs([(a, a[1:M_]) for a in two_dim_var])
            fde = dx * (u[1:M_] - u0[1:M_]) + simplify(dt * (fui1j1 - fuij1)) - simplify(dx * dt * S)
        elif direction == -1:
            fui1j1 = flux.subs([(a, a[1:M_]) for a in two_dim_var])
            fuij1 = flux.subs([(a, a[0:M_ - 1]) for a in two_dim_var])
            S = source.subs([(a, a[0:M_ - 1]) for a in two_dim_var])
            fde = dx * (u[0:M_ - 1] - u0[0:M_ - 1]) + simplify(dt * (fui1j1 - fuij1)) - simplify(dx * dt * S)
        else:
            raise ValueError(f"Unimplemented direction {direction}!")
    return fde


def semi_descritize(diff_var,
                    flux,
                    source,
                    two_dim_var,
                    M,
                    scheme='TVD1',
                    a0=None,
                    a1=None,
                    dx=None):
    M_ = M + 1  # for pretty printer of M in slice

    if a0 is None:
        a0 = Para('ajp12')
    if a1 is None:
        a1 = Para('ajm12')

    if dx is None:
        dx = Para('dx')
    else:
        if is_number(dx):
            dx = dx
        else:
            raise TypeError(f'Input dx is not number!')

    u = diff_var
    if scheme == 'TVD2':
        # j=1
        # f(u[2])
        fu2 = flux.subs([(var, var[2]) for var in two_dim_var])
        # f(u[0])=f(2*uL-u[1])
        fu0 = flux.subs([(var, var[0]) for var in two_dim_var])
        # S(u[1])
        Su1 = source.subs([(var, var[1]) for var in two_dim_var])
        ode_rhs1 = -simplify((fu2 - fu0) / (2 * dx)) \
                   + simplify((a0[0] * (u[2] - u[1]) - a1[0] * (u[1] - u[0])) / (2 * dx)) \
                   + simplify(Su1)

        # j=M-1
        # f(u[M])=f(2*uR-u[M-1])
        fum = flux.subs([(var, var[M]) for var in two_dim_var])
        # f(u[M-2])
        fum2 = flux.subs([(var, var[M - 2]) for var in two_dim_var])
        # S(u[M-1])
        SuM1 = source.subs([(var, var[M - 1]) for var in two_dim_var])
        ode_rhs3 = -simplify((fum - fum2) / (2 * dx)) \
                   + simplify((a0[-1] * (u[M] - u[M - 1]) - a1[-1] * (u[M - 1] - u[M - 2])) / (2 * dx)) \
                   + simplify(SuM1)

        # 2<=j<=M-2
        def ujprime(U: IdxVar, v: int):
            # for given u_j,
            # returns
            # u^+_{j+1/2} case v==0,
            # u^-_{j+1/2} case 1,
            # u^+_{j-1/2} case 2,
            # u^-_{j-1/2} case 3
            if not isinstance(U.index, slice):
                raise TypeError("Index of IdxVar must be slice object")
            start = U.index.start
            stop = U.index.stop
            step = U.index.step
            U = U.symbol0
            Ux = iVar(U.name + 'x')

            # u_j
            Uj = U[start:stop:step]
            # (u_x)_j
            Uxj = Ux[start:stop:step]
            # u_{j+1}
            Ujp1 = U[start + 1:stop + 1:step]
            # (u_x)_{j+1}
            Uxjp1 = Ux[start + 1:stop + 1:step]
            # u_{j-1}
            Ujm1 = U[start - 1:stop - 1:step]
            # (u_x)_{j-1}
            Uxjm1 = Ux[start - 1:stop - 1:step]

            if v == 0:
                return Ujp1 - dx / 2 * Uxjp1
            elif v == 1:
                return Uj + dx / 2 * Uxj
            elif v == 2:
                return Uj - dx / 2 * Uxj
            elif v == 3:
                return Ujm1 + dx / 2 * Uxjm1
            else:
                raise ValueError("v=0 or 1 or 2 or 3!")

        # j\in [2:M_-2]
        Suj = source.subs([(var, var[2:M_ - 2]) for var in two_dim_var])
        Hp = (flux.subs([(var, ujprime(var[2:M_ - 2], 0)) for var in two_dim_var]) +
              flux.subs([(var, ujprime(var[2:M_ - 2], 1)) for var in two_dim_var])) / 2 \
             - a0[2:M_ - 2] / 2 * (ujprime(u[2:M_ - 2], 0) - ujprime(u[2:M_ - 2], 1))
        Hm = (flux.subs([(var, ujprime(var[2:M_ - 2], 2)) for var in two_dim_var]) +
              flux.subs([(var, ujprime(var[2:M_ - 2], 3)) for var in two_dim_var])) / 2 \
             - a1[2:M_ - 2] / 2 * (ujprime(u[2:M_ - 2], 2) - ujprime(u[2:M_ - 2], 3))
        ode_rhs2 = -simplify(Hp - Hm) / dx + Suj

        theta = Para('theta')
        ux = iVar(u.name + 'x')
        minmod_flag = Para('minmod_flag_of_' + ux.name)
        minmod_rhs = ux[1:M_ - 1] - switch(theta * (u[1:M_ - 1] - u[0:M_ - 2]) / dx,
                                           (u[2:M_] - u[0:M_ - 2]) / (2 * dx),
                                           theta * (u[2:M_] - u[1:M_ - 1]) / dx,
                                           0,
                                           minmod_flag)

        return {'Ode': [(ode_rhs1, u[1]), (ode_rhs2, u[2:M_ - 2]), (ode_rhs3, u[M - 1])],
                'Eqn': [minmod_rhs, ux[0], ux[M]]}
    elif scheme == 'TVD1':
        # 1<=j<=M-1
        # f(u[j+1])
        fu1 = flux.subs([(var, var[2:M_]) for var in two_dim_var])
        # f(u[j-1])
        fu2 = flux.subs([(var, var[0:M_ - 2]) for var in two_dim_var])
        # S(u[j])
        Su = source.subs([(var, var[1:M_ - 1]) for var in two_dim_var])
        ode_rhs = -simplify((fu1 - fu2) / (2 * dx)) \
                  + simplify((a0 * (u[2:M_] - u[1:M_ - 1]) - a1 * (u[1:M_ - 1] - u[0:M_ - 2])) / (2 * dx)) \
                  + simplify(Su)
        return {'Ode': (ode_rhs, u[1:M_ - 1])}
