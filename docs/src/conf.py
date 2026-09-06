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
exclude_patterns = ['_static/**', '_templates/**']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# ``myst_parser`` parses .md pages but treats ``$...$`` / ``$$...$$``
# blocks as literal text unless ``dollarmath`` is enabled. ``amsmath``
# lets authors use ``\begin{align}``-style environments inside
# ``$$...$$``; ``colon_fence`` supports the ``:::{math}``/``:::``
# fenced form used across the Solverz Cookbook.
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'colon_fence',
]

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
    # Link orange is darker than logo orange to keep body links readable.
    "color-brand-primary": "#A74418",
    "color-brand-content": "#A74418",
    "color-brand-visited": "#805041",
    "color-foreground-primary": "#223746",
    "color-foreground-secondary": "#526573",
    "color-foreground-muted": "#617480",
    "color-foreground-border": "#8396A1",
    "color-background-primary": "#FFFFFF",
    "color-background-secondary": "#F5F7F8",
    "color-background-hover": "#EDF2F4",
    "color-background-border": "#DFE7EB",
    "color-sidebar-background": "#FFFFFF",
    "color-sidebar-background-border": "#DFE7EB",
    "color-sidebar-link-text": "#425B6B",
    "color-sidebar-link-text--top-level": "#164162",
    "color-sidebar-caption-text": "#617480",
    "color-sidebar-item-background--current": "#FFF2E9",
    "color-sidebar-item-background--hover": "#F5F7F8",
    "color-sidebar-item-expander-background--hover": "#EDF2F4",
    "color-sidebar-search-background": "#F5F7F8",
    "color-sidebar-search-background--focus": "#FFFFFF",
    "color-api-name": "#164162",
    "color-api-pre-name": "#526573",
    "color-api-background": "#F5F7F8",
    "color-api-background-hover": "#EDF2F4",
    "color-api-keyword": "#A74418",
    "color-highlight-on-target": "#FFF0CC",
    "color-highlighted-background": "#FFE4BE",
    "color-admonition-background": "#F8FAFB",
    "color-admonition-title-background--note": "#E9F4F4",
    "color-admonition-title--note": "#126D76",
    "color-admonition-title-background--seealso": "#EEF3F8",
    "color-admonition-title--seealso": "#164162",
    "color-code-background": "#F6F8FA",
    "color-code-foreground": "#223746",
    "font-stack": '-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif',
    "font-stack--monospace": '"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace',
    "code-font-size": "0.875rem",
    "admonition-font-size": "0.95rem",
    "solverz-accent": "#F96A20",
    "solverz-heading": "#164162",
    "solverz-surface": "#F7F9FA",
    "solverz-soft-accent": "#FFF2E9",
    "solverz-button-background": "#164162",
    "solverz-button-foreground": "#FFFFFF",
}

html_theme_options = {
    "light_logo": "brand/logo-horizontal-light.png",
    "dark_logo": "brand/logo-horizontal-dark.png",
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/smallbunnies/Solverz/",
    "source_branch": "main",
    "source_directory": "docs/src/",
    "light_css_variables": common_theme_variables,
    "dark_css_variables": {
        **common_theme_variables,
        "color-brand-primary": "#FFAE76",
        "color-brand-content": "#FFAE76",
        "color-brand-visited": "#E5BAA7",
        "color-foreground-primary": "#E5EDF2",
        "color-foreground-secondary": "#B5C6D0",
        "color-foreground-muted": "#99AFBD",
        "color-foreground-border": "#667F90",
        "color-background-primary": "#102433",
        "color-background-secondary": "#172F40",
        "color-background-hover": "#203C50",
        "color-background-border": "#2D4658",
        "color-sidebar-background": "#142A3A",
        "color-sidebar-background-border": "#2D4658",
        "color-sidebar-link-text": "#C2D0D9",
        "color-sidebar-link-text--top-level": "#E5EDF2",
        "color-sidebar-caption-text": "#99AFBD",
        "color-sidebar-item-background--current": "#273C4B",
        "color-sidebar-item-background--hover": "#203C50",
        "color-sidebar-item-expander-background--hover": "#203C50",
        "color-sidebar-search-background": "#102433",
        "color-sidebar-search-background--focus": "#1C3749",
        "color-api-name": "#8DD7DF",
        "color-api-pre-name": "#B5C6D0",
        "color-api-background": "#172F40",
        "color-api-background-hover": "#203C50",
        "color-api-keyword": "#FFAE76",
        "color-highlight-on-target": "#4E401D",
        "color-highlighted-background": "#674321",
        "color-admonition-background": "#172F40",
        "color-admonition-title-background--note": "#183E48",
        "color-admonition-title--note": "#8DD7DF",
        "color-admonition-title-background--seealso": "#203C50",
        "color-admonition-title--seealso": "#C2D9E7",
        "color-code-background": "#142B3C",
        "color-code-foreground": "#E5EDF2",
        "solverz-heading": "#F1F6F8",
        "solverz-surface": "#172F40",
        "solverz-soft-accent": "#273C4B",
        "solverz-button-background": "#FFAE76",
        "solverz-button-foreground": "#142A3A",
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

html_title = 'Solverz documentation'
html_baseurl = os.environ.get('READTHEDOCS_CANONICAL_URL', 'https://docs.solverz.org/')
html_context = {
    'solverz_description': 'General-purpose modeling and simulation in Python. Define symbolic equations, generate numerical functions, and solve your models with Solverz.',
    'solverz_social_image': html_baseurl.rstrip('/') + '/_static/brand/social-cover.png',
}
pygments_style = 'friendly'
pygments_dark_style = 'native'

# Reuse the approved square mark rather than maintaining another icon design.
latex_logo = '_static/brand/logo-stacked.png'
html_favicon = '_static/brand/symbol-square.png'


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
