.. _reference:

=============
API Reference
=============

Functions
---------

.. autoclass:: Solverz.sym_algebra.functions.sin

.. autoclass:: Solverz.sym_algebra.functions.cos

.. autoclass:: Solverz.sym_algebra.functions.exp

.. autoclass:: Solverz.sym_algebra.functions.Abs

.. autoclass:: Solverz.sym_algebra.functions.Sign

.. autoclass:: Solverz.sym_algebra.functions.AntiWindUp

.. autoclass:: Solverz.sym_algebra.functions.Min

.. autoclass:: Solverz.sym_algebra.functions.MatVecMul

.. autoclass:: Solverz.sym_algebra.functions.Saturation

.. autoclass:: Solverz.sym_algebra.functions.heaviside

.. autoclass:: Solverz.sym_algebra.functions.ln

Equations
---------

.. autoclass:: Solverz.equation.eqn.Eqn

.. autoclass:: Solverz.equation.eqn.Ode

.. autoclass:: Solverz.equation.eqn.LoopEqn

.. autoclass:: Solverz.equation.eqn.LoopOde

.. autoclass:: Solverz.equation.eqn.IndexSet
   :members:

.. autofunction:: Solverz.equation.eqn.Idx

.. autofunction:: Solverz.equation.eqn.Sum

Utilities
---------

.. autofunction:: Solverz.utilities.miscellaneous.derive_incidence_matrix

Solvers
-------

AE solver
=========

.. autofunction:: Solverz.solvers.nlaesolver.nr.nr_method

.. autofunction:: Solverz.solvers.nlaesolver.cnr.continuous_nr

.. autofunction:: Solverz.solvers.nlaesolver.lm.lm

.. autofunction:: Solverz.solvers.nlaesolver.sicnm.sicnm

FDAE solver
===========

.. autofunction:: Solverz.solvers.fdesolver.fdae_solver

DAE solver
==========

.. autofunction:: Solverz.solvers.daesolver.beuler.backward_euler

.. autofunction:: Solverz.solvers.daesolver.trapezoidal.implicit_trapezoid

The legacy ``Rodas`` is deprecated; the Rosenbrock methods of the :ref:`integrator <integrator>` replace it.

.. autofunction:: Solverz.solvers.daesolver.rodas.Rodas

.. autofunction:: Solverz.solvers.daesolver.radau.Radau

.. autofunction:: Solverz.solvers.daesolver.pe.PE

.. autofunction:: Solverz.solvers.daesolver.adams_bdf.AdamsBDF

.. autofunction:: Solverz.solvers.daesolver.ode15s.ode15s

Integrator
==========

See :ref:`integrating a DAE <integrator>` and :ref:`adding an algorithm <integrator_adding_algorithms>`.

.. autofunction:: Solverz.integrator.solve

.. autofunction:: Solverz.integrator.init

.. autoclass:: Solverz.integrator.Integrator
   :members: step, solve, interp, terminate, model_modified

.. autoclass:: Solverz.integrator.IntegratorOptions
   :members: from_opt

Methods:

.. autoclass:: Solverz.integrator.Rodas3

.. autoclass:: Solverz.integrator.Rodas4

.. autoclass:: Solverz.integrator.Rodasp

.. autoclass:: Solverz.integrator.Rodas5P

.. autoclass:: Solverz.integrator.ImplicitEuler

.. autoclass:: Solverz.integrator.Trapezoid

The author contract:

.. autoclass:: Solverz.integrator.Algorithm
   :members: perform_step, alloc, interpolant, addsteps, reset_history, controller, initial_dt, __call__

.. autoexception:: Solverz.integrator.StepFailure

.. autoclass:: Solverz.integrator.Rosenbrock
   :members: from_scheme

.. autoclass:: Solverz.integrator.RosenbrockTableau
   :members: from_hairer

.. autofunction:: Solverz.integrator.testing.check_algorithm

Callbacks:

.. autoclass:: Solverz.integrator.ContinuousCallback

.. autoclass:: Solverz.integrator.DiscreteCallback

.. autofunction:: Solverz.integrator.preset_time_callback

Step-size controllers:

.. autoclass:: Solverz.integrator.Controller

.. autoclass:: Solverz.integrator.IController

.. autoclass:: Solverz.integrator.PIController

.. autoclass:: Solverz.integrator.LegacyRodasController
