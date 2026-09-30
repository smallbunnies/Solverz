"""The integrator core: one stepping loop that owns step control, saving,
events and the linear algebra, and one ``perform_step`` per algorithm.

``__all__`` is explicit, so that ``from Solverz.integrator import *`` binds
no submodule name.
"""
from Solverz.integrator.options import IntegratorOptions
from Solverz.integrator.algorithm import Algorithm, StepFailure

__all__ = ['IntegratorOptions',
           'Algorithm', 'StepFailure']
