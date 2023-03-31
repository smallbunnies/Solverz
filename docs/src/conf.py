# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import inspect
import os
import sys

import sympy

sys.path = ['ext'] + sys.path
sys.path.extend(['..\\..\\Solverz'])
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Solverz'
copyright = '2023, Ruizhi Yu'
author = 'Ruizhi Yu'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx_math_dollar', 'sphinx.ext.mathjax', 'numpydoc',
              'sphinx_reredirects', 'sphinx_copybutton',
              'sphinx.ext.graphviz', 'matplotlib.sphinxext.plot_directive',
              'myst_parser', 'convert-svg-to-pdf', 'sphinx.ext.intersphinx', ]  # 'sphinx.ext.linkcode'

# To stop docstrings inheritance.
autodoc_inherit_docstrings = False

# Sphinx是一个文档生成器，可以将Markdown或reStructuredText等文本格式转化为HTML、PDF等格式的文档。而MathJax是一个用于渲染数学公式的JavaScript库，它可以帮助将LaTeX或MathML格式的数学公式渲染为高质量的矢量图形。
# 虽然Sphinx本身提供了一些对数学公式的支持，但其渲染效果不如MathJax优秀。因此，为了获得更好的数学公式渲染效果，使用MathJax插件可以帮助Sphinx在生成文档时自动渲染数学公式，从而提高文档的质量和可读性。

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

mathjax3_config = {
    "tex": {
        "inlineMath": [['\\(', '\\)']],
        "displayMath": [["\\[", "\\]"]],
    }
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != 'py':
        return

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(sympy.__file__))
    return blobpath + fn + linespec
