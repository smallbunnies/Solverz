"""A rendered model against the same model inline.

In the legacy-compatible configuration each of the two is byte-equal to its
own legacy Rodas run, per model and scheme; ``test_compat_parity.py`` asserts
this over its whole matrix, rendered and inline variants included, and it is
not repeated here.

Between the rendered and the inline model byte equality is not required,
because Numba and NumPy evaluate ``**`` and transcendental functions with
different routines. The criterion in the default configuration, the one used
wherever a rendered model is compared with its inline form, the conformance
kit included, is ``rendered_matches_inline`` of ``Solverz.integrator.testing``:
the numbers of accepted steps differ by at most 2, and the saved rows by at
most ``10 * rtol``.
"""
import numpy as np
import pytest

from Solverz.integrator import Rosenbrock
from Solverz.integrator.testing import rendered_matches_inline
from Solverz.solvers.option import Opt

from tests.integrator.test_legacy_transcription import SCHEMES

pytestmark = pytest.mark.i5


@pytest.mark.parametrize('scheme', SCHEMES)
def test_rendered_and_inline_dae_test(model, scheme):
    inline, y0 = model('dae_test')
    rendered, _ = model('dae_test', 'rendered')
    tspan = np.linspace(0, 20, 201)
    rtol = 1e-6
    alg = Rosenbrock.from_scheme(scheme)
    si = alg(inline, tspan, y0, Opt(rtol=rtol, atol=1e-8))
    sr = alg(rendered, tspan, y0, Opt(rtol=rtol, atol=1e-8))
    assert si.stats.ret == sr.stats.ret == 'success'
    dsteps, dY, ok = rendered_matches_inline(sr, si, rtol)
    print(f"{scheme}: accepted steps {sr.stats.nstep} rendered, {si.stats.nstep} inline, "
          f"difference {dsteps}; max|dY| = {dY:.3e}")
    assert ok, (dsteps, dY)
