"""
Radau IIA(5) tableau and Hairer-Wanner complex-basis transform
constants. Direct port of SciML OrdinaryDiffEqFIRK
``RadauIIA5Tableau`` (firk_tableaus.jl, lines ~73-118).
"""

import numpy as np

# Stage abscissae (c3 = 1 implicit in the perform-step code).
SQRT6 = np.sqrt(6.0)
c1 = (4 - SQRT6) / 10
c2 = (4 + SQRT6) / 10
c3 = 1.0

# Real-form transformation T (T32 = 1, T33 = 0 implicit in the
# SciML code via `z3 = T31*w1 + w2`).
T11 = 9.1232394870892942792e-2
T12 = -0.14125529502095420843
T13 = -3.0029194105147424492e-2
T21 = 0.24171793270710701896
T22 = 0.20412935229379993199
T23 = 0.38294211275726193779
T31 = 0.96604818261509293619
T32 = 1.0
T33 = 0.0

T_HW = np.array([[T11, T12, T13],
                 [T21, T22, T23],
                 [T31, T32, T33]])

TI11 = 4.325579890063155351
TI12 = 0.33919925181580986954
TI13 = 0.54177053993587487119
TI21 = -4.1787185915519047273
TI22 = -0.32768282076106238708
TI23 = 0.47662355450055045196
TI31 = -0.50287263494578687595
TI32 = 2.5719269498556054292
TI33 = -0.59603920482822492497

TI_HW = np.array([[TI11, TI12, TI13],
                  [TI21, TI22, TI23],
                  [TI31, TI32, TI33]])

# Eigenvalues of A^{-1}: one real (gamma), one complex pair (alpha ± i*beta).
# Derived in SciML's tableau function:
_cbrt9 = np.cbrt(9.0)
_gamma_prime = (6.0 + _cbrt9 * (_cbrt9 - 1.0)) / 30.0     # eigval of A
_alpha_prime = (12.0 - _cbrt9 * (_cbrt9 - 1.0)) / 60.0    # eigval of A
_beta_prime = _cbrt9 * (_cbrt9 + 1.0) * np.sqrt(3.0) / 60.0
_scale = _alpha_prime ** 2 + _beta_prime ** 2

GAMMA = 1.0 / _gamma_prime
ALPHA = _alpha_prime / _scale
BETA = _beta_prime / _scale

# Embedded error coefficients.
E1 = -(13 + 7 * SQRT6) / 3
E2 = (-13 + 7 * SQRT6) / 3
E3 = -1.0 / 3.0

# Butcher matrix A (kept for reference / external use).
A = np.array([
    [(88 - 7 * SQRT6) / 360, (296 - 169 * SQRT6) / 1800, (-2 + 3 * SQRT6) / 225],
    [(296 + 169 * SQRT6) / 1800, (88 + 7 * SQRT6) / 360, (-2 - 3 * SQRT6) / 225],
    [(16 - SQRT6) / 36, (16 + SQRT6) / 36, 1 / 9],
])
b = A[-1, :].copy()


def get_butcher():
    return A.copy(), b.copy(), np.array([c1, c2, c3])


def get_hw_constants():
    return GAMMA, ALPHA, BETA, T_HW.copy(), TI_HW.copy()


def get_error_coeffs():
    return E1, E2, E3
