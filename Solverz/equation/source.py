"""Provenance stamping for Solverz equations.

A model fragment assembled by a reusable component (e.g. a SolMuseum or
SolPSDyn building block) can tag every equation it contributes with a
``source`` dict ``{'component', 'package', 'version'}``. The module
printer renders that provenance into the generated module's docstrings.
Solverz core never imports the downstream packages; the version flows in
as data via this helper.
"""
from __future__ import annotations

from Solverz.equation.eqn import Eqn


def stamp_source(model, *, component, package, version, overwrite=False):
    """Set ``.source`` on every equation attribute of ``model``.

    Walks the model's attributes (where freshly built ``Eqn`` / ``Ode`` /
    ``LoopEqn`` / ``LoopOde`` objects live before ``create_instance``) and
    stamps each with ``{'component', 'package', 'version'}``.

    Parameters
    ----------
    model : Solverz.Model
    component, package, version : str
        Provenance fields. ``component`` is the element/model type (e.g.
        ``'gt'``, ``'eps_network'``), ``package`` the distribution (e.g.
        ``'SolMuseum'``), ``version`` its version string.
    overwrite : bool, default False
        When False, equations already carrying a (more specific) source
        are left untouched, so a composite block does not clobber the
        stamps of sub-fragments it aggregates.

    Returns
    -------
    model : the same object, for chaining.
    """
    src = {'component': str(component),
           'package': str(package),
           'version': str(version)}
    for value in vars(model).values():
        if isinstance(value, Eqn):  # covers Ode/LoopEqn/LoopOde (all subclass Eqn)
            if overwrite or getattr(value, 'source', None) is None:
                value.source = dict(src)
    return model


def format_source(source):
    """Return a one-line ``' — <package> <version> / <component>'`` suffix
    for a source dict, or ``''`` when ``source`` is falsy."""
    if not source:
        return ''
    return (f" — {source.get('package', '?')} "
            f"{source.get('version', '?')} / {source.get('component', '?')}")
