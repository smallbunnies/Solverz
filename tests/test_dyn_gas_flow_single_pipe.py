import numpy as np
import pandas as pd

from Solverz import idx, Var, Para, as_Vars, HyperbolicPde, fdae_solver, Eqn, Param, IdxParam, FDAE, made_numerical, \
    parse_dae_v

results = pd.read_excel('instances/dynamic_gas_flow_single_pipe.xlsx',
                        sheet_name=None,
                        engine='openpyxl'
                        )

Pie = Var('Pie', value=np.linspace(9, 8.728, 11) * 1e6)
q = Var('q', value=7.3125 * np.ones((11,)))
va = Para('va', value=340)
D = Para('D', value=0.5)
S = Para('S', value=np.pi * (D.value / 2) ** 2)
lam = Para('lam', 0.3)
pb = Para('pb', 9e6)
qb = Para('qb', 7.3125)
M = idx('M')
L = 3000

gas_pde1 = HyperbolicPde('mass conservation',
                         diff_var=Pie,
                         flux=va ** 2 / S * q,
                         two_dim_var=[Pie, q])
gas_pde2 = HyperbolicPde('momentum conservation',
                         diff_var=q,
                         flux=S * Pie,
                         source=-lam * va ** 2 * q ** 2 / (2 * D * S * Pie),
                         two_dim_var=[Pie, q])

ae1 = gas_pde1.finite_difference()
ae2 = gas_pde2.finite_difference()
ae3 = Eqn(name='boundary condition Pie', eqn=Pie[0] - pb)
ae4 = Eqn(name='boundary condition q', eqn=q[M] - qb)

gas_FDE = FDAE([ae1, ae2, ae3, ae4], name='gas FDE', nstep=1)
u0 = as_Vars([Pie, q])
gas_FDE.update_param(u0.derive_alias('_tag_0'))

gas_FDE.param_initializer('M', IdxParam('M', value=np.array(10)))
gas_FDE.param_initializer('dx', Param('dx', value=np.array(300)))
gas_FDE.param_initializer('dt', Param('dt', value=5))

ngas = made_numerical(gas_FDE, u0)
T, u, stats = fdae_solver(ngas, [0, 3600], u0.array, 5, tol=1e-3)
u = parse_dae_v(u, u0.a)
ngas_sparse, code = made_numerical(gas_FDE, u0, sparse=True, output_code=True)

T1, u1, stats1 = fdae_solver(ngas_sparse, [0, 3600], u0.array, 5, tol=1e-3)
u1 = parse_dae_v(u1, u0.a)


def test_fde_solver():
    pout = np.asarray(results['results']['pout'])
    qin = np.asarray(results['results']['qin'])

    # if matrix container is cvxopt.spmatrix
    # assert np.mean(abs(pout - u['Pie'][:, -1] / 1e6)) < 5e-6
    # assert np.mean(abs(qin - u['q'][:, 0])) < 1.2e-4

    # if matrix container is scipy.sparse.csc_array
    assert np.mean(abs(pout - u['Pie'][:, -1] / 1e6)) < 1e-10
    assert np.mean(abs(qin - u['q'][:, 0])) < 1e-7
    assert np.mean(abs(pout - u1['Pie'][:, -1] / 1e6)) < 1e-10
    assert np.mean(abs(qin - u1['q'][:, 0])) < 1e-7
