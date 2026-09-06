:hide-toc:

.. _documentation:

.. module:: solverz

Solverz documentation
=====================

`Solverz <https://github.com/smallbunnies/Solverz>`_ is a general-purpose modeling and simulation toolkit for Python. Define symbolic equations, generate numerical functions, and solve algebraic, differential, and finite-difference models.

Start with :doc:`installation <install>` and the :ref:`introductory example <intro>`. The :ref:`modeling guide <gettingstarted>` explains variables, parameters, and equations. The :doc:`API reference <reference/index>` describes the modeling objects and solvers.

Find worked examples in the `Solverz Cookbook <https://cookbook.solverz.org/latest/>`_. Advanced topics include :doc:`matrix calculus <matrix_calculus>`, :doc:`indexed equations <loopeqn>`, and :doc:`custom functions and code generation <advanced>`.

.. The HTML homepage uses partials/solverz-home.html. Keep this description and
   the toctrees available to search indexing and non-HTML documentation builds.

.. toctree::
   :hidden:
   :caption: Start here

   install.md
   intro.md
   gettingstart.md

.. toctree::
   :hidden:
   :caption: Modeling tools

   advanced.md
   matrix_calculus.md
   extend_matrix_calculus.md
   loopeqn.md
   loopeqn_translator.md

.. toctree::
   :hidden:
   :caption: Reference and community

   reference/index.rst
   release_notes.md
   gethelp.md
   contributing.md
   brand.md
