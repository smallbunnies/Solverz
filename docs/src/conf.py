# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import inspect
import os
import sys
import time

import sympy

sys.path.extend(['..\\..\\Solverz'])
# This command is for linux (Ubuntu 22.04.2 LTS), which cannot recognise relative path of Solverz library.
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../ext'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Solverz'
copyright = f'{time.localtime().tm_year}, Ruizhi Yu'
author = 'Ruizhi Yu'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest',
              'sphinx_math_dollar', 'sphinx.ext.mathjax', 'numpydoc',
              'sphinx_reredirects', 'sphinx_copybutton',
              'sphinx.ext.graphviz', 'sphinxcontrib.jquery',
              'matplotlib.sphinxext.plot_directive', 'myst_parser',
              'convert-svg-to-pdf', 'sphinx.ext.intersphinx', ]  # 'sphinx.ext.linkcode'

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
        'packages': {'[+]': ['physics']}
    },
    'loader': {'load': ['[tex]/physics']},
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_static_path = ['_static']

html_theme = 'furo'

common_theme_variables = {
    # Main "SymPy green" colors. Many things uses these colors.
    "color-brand-primary": "#52833A",
    "color-brand-content": "#307748",

    # The left sidebar.
    "color-sidebar-background": "#3B5526",
    "color-sidebar-background-border": "var(--color-background-primary)",
    "color-sidebar-link-text": "#FFFFFF",
    "color-sidebar-brand-text": "var(--color-sidebar-link-text--top-level)",
    "color-sidebar-link-text--top-level": "#FFFFFF",
    "color-sidebar-item-background--hover": "var(--color-brand-primary)",
    "color-sidebar-item-expander-background--hover": "var(--color-brand-primary)",

    "color-link-underline--hover": "var(--color-link)",
    "color-api-keyword": "#000000bd",
    "color-api-name": "var(--color-brand-content)",
    "color-api-pre-name": "var(--color-brand-content)",
    "api-font-size": "var(--font-size--normal)",
    "color-foreground-secondary": "#53555B",

    # TODO: Add the other types of admonitions here if anyone uses them.
    "color-admonition-title-background--seealso": "#CCCCCC",
    "color-admonition-title--seealso": "black",
    "color-admonition-title-background--note": "#CCCCCC",
    "color-admonition-title--note": "black",
    "color-admonition-title-background--warning": "var(--color-problematic)",
    "color-admonition-title--warning": "white",
    "admonition-font-size": "var(--font-size--normal)",
    "admonition-title-font-size": "var(--font-size--normal)",

    # Note: this doesn't work. If we want to change this, we have to set
    # it as the .highlight background in custom.css.
    "color-code-background": "hsl(80deg 100% 95%)",

    "code-font-size": "var(--font-size--small)",
    "font-stack--monospace": 'DejaVu Sans Mono,"SFMono-Regular",Menlo,Consolas,Monaco,Liberation Mono,Lucida Console,monospace;'
}

html_theme_options = {
    "light_css_variables": common_theme_variables,
    # The dark variables automatically inherit values from the light variables
    "dark_css_variables": {
        **common_theme_variables,
        "color-brand-primary": "#33CB33",
        "color-brand-content": "#1DBD1D",

        "color-api-keyword": "#FFFFFFbd",
        "color-api-overall": "#FFFFFF90",
        "color-api-paren": "#FFFFFF90",

        "color-sidebar-item-background--hover": "#52833A",
        "color-sidebar-item-expander-background--hover": "#52833A",
        # This is the color of the text in the right sidebar
        "color-foreground-secondary": "#9DA1AC",

        "color-admonition-title-background--seealso": "#555555",
        "color-admonition-title-background--note": "#555555",
        "color-problematic": "#B30000",
    },
    # See https://pradyunsg.me/furo/customisation/footer/
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/smallbunnies/Solverz",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

html_css_files = ['custom.css']

html_domain_indices = ['py-modindex']

# Solverz logo on title page
html_logo = '_static/sympylogo.png'
latex_logo = '_static/sympylogo_big.png'
html_favicon = '../_build/logo/sympy-notailtext-favicon.ico'


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
