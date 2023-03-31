.. _sas_module:


Contents
========

.. toctree::
    :maxdepth: 3

    numerical.rst
    algebra.rst


Performance improvements
========================

On queries that involve symbolic coefficients, logical inference is used. Work on
improving satisfiable function (sympy.logic.inference.satisfiable) should result
in notable speed improvements.

Logic inference used in one ask could be used to speed up further queries, and
current system does not take advantage of this. For example, a truth maintenance
system (https://en.wikipedia.org/wiki/Truth_maintenance_system) could be implemented.
