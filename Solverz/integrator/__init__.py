"""The integrator core: one stepping loop that owns step control, saving,
events and the linear algebra, and one ``perform_step`` per algorithm.

``__all__`` is explicit, so that ``from Solverz.integrator import *`` binds
no submodule name.
"""
from Solverz.integrator.integrator import solve, init, Integrator
from Solverz.integrator.options import IntegratorOptions
from Solverz.integrator.algorithm import Algorithm, StepFailure
from Solverz.integrator.rosenbrock import Rosenbrock, RosenbrockTableau, Rodas3, Rodas4, Rodasp, Rodas5P
from Solverz.integrator.implicit_euler import ImplicitEuler
from Solverz.integrator.trapezoid import Trapezoid
from Solverz.integrator.callbacks import ContinuousCallback, DiscreteCallback, preset_time_callback
from Solverz.integrator.controllers import Controller, IController, PIController, LegacyRodasController

__all__ = ['solve', 'init', 'Integrator', 'IntegratorOptions',
           'Algorithm', 'StepFailure',
           'Rosenbrock', 'RosenbrockTableau', 'Rodas3', 'Rodas4', 'Rodasp', 'Rodas5P',
           'ImplicitEuler', 'Trapezoid',
           'ContinuousCallback', 'DiscreteCallback', 'preset_time_callback',
           'Controller', 'IController', 'PIController', 'LegacyRodasController']
