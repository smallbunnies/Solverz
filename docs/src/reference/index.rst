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

.. autofunction:: Solverz.solvers.daesolver.rodas.Rodas

.. autofunction:: Solverz.solvers.daesolver.ode15s.ode15s
