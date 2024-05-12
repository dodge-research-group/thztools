# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# The importlib.metadata library is used to import the version number from
# PyPI using OPTION 2 below. Not needed for OPTION 1.
# import importlib.metadata

# The runpy package is used to import the version number from
# __about__.py using OPTION 1 below. Not needed for OPTION 2.
import math
import os
import runpy
import sys
from pathlib import Path

import tomli  # Can use tomllib for python >= 3.11

sys.path.insert(0, (Path(__file__).parents[2] / "src").resolve().as_posix())

# -- Project information -----------------------------------------------------

with (Path(__file__).parents[2] / "pyproject.toml").open("rb") as f:
    pkg_info = tomli.load(f)
project = pkg_info["project"]["name"]

author = "Steve Dodge"
pkg_creation_year = 2023
project_copyright = f"{pkg_creation_year} - present, {author}"

# Version number for docs
# ========================
# OPTION 1: Version number listed in GitHib
version = runpy.run_path(
    Path(__file__).parents[2] / "src" / "thztools" / "__about__.py"
)["__version__"]
#
# OPTION 2: Version number listed in PyPI
# version = importlib.metadata.version(project)
# release = version

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "nbsphinx",
    "numpydoc",
    "sphinx_design",
    "sphinx_gallery.load_style",
    "sphinx_tabs.tabs",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# Show successful tests
doctest_show_successes = True

# Setup code for doctests
doctest_global_setup = """
import numpy as np
import thztools as thz
rng = np.random.default_rng(0)
"""

# -----------------------------------------------------------------------------
# nbsphinx configuration
# -----------------------------------------------------------------------------

nbsphinx_allow_errors = True

# Set PYTHONPATH to local src directory (otherwise installs from PyPI)
src = os.path.abspath("../../src")
os.environ["PYTHONPATH"] = str(src)

# -----------------------------------------------------------------------------
# NumPy extensions
# -----------------------------------------------------------------------------

# Make numpydoc to generate plots for example sections
numpydoc_use_plots = True

numpydoc_class_members_toctree = True
numpydoc_attributes_as_param_list = False

# Cross-reference parameter types when True
numpydoc_xref_param_type = False

# Report warnings for all validation checks
numpydoc_validation_checks = {"all", "SA01", "ES01", "RT02", "EX01", "PR01"}

# Ensure all our internal links work
nitpicky = True

# -----------------------------------------------------------------------------
# Matplotlib plot_directive options (adapted from SciPy docs)
# -----------------------------------------------------------------------------

# Display source code for Matplotlib plot directives
plot_include_source = True
plot_formats = [("png", 100), "pdf"]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_pre_code = """
import numpy as np
"""
plot_apply_rcparams = True

phi = (math.sqrt(5) + 1) / 2

font_size = 13 * 72 / 96.0  # 13 px

plot_rcparams = {
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "figure.figsize": (3 * phi, 3),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}

# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------

autodoc_typehints = "none"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------

intersphinx_mapping = {
    "numpy": ("https://numpy.org/devdocs", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["./_static"]

html_theme_options = {
    "logo": {
        "image_light": "thztools_logo.svg",
        "image_dark": "thztools_logo_dark.svg",
    },
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect (required)
            "url": "https://github.com/dodge-research-group/thztools",
            # Icon class (if "type": "fontawesome"), or path to local image
            # (if "type": "local")
            "icon": "fab fa-github-square",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "thztoolsdoc"

# -- Options for LaTeX output ------------------------------------------------

# The master toctree document.
master_doc = "index"

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    #
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    #
    "preamble": "",
    # Latex figure (float) alignment
    #
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "thztools.tex",
        "thztools Documentation",
        "thztools",
        "manual",
    ),
]
