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
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("./_ext"))


# -- Project information -----------------------------------------------------

project = "smash"
copyright = "2022, Francois Colleoni"
author = "Francois Colleoni"

# The full version, including alpha/beta/rc tags
release = "0.2.2"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinxcontrib.bibtex",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx_design",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autosummary_accessors",
    "optimize_directive",
]


autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 1

pygments_style = "sphinx"

numpydoc_show_class_members = True

autosummary_generate = True  # Turn on sphinx.ext.autosummary

autodoc_typehints = "none"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

default_role = "autolink"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_favicon = "_static/favicon.ico"

html_theme = "pydata_sphinx_theme"

html_last_updated_fmt = "%b %d, %Y"

html_theme_options = {
    "gitlab_url": "https://gitlab.irstea.fr/francois.colleoni/smash/",
    "collapse_navigation": True,
    "logo": {
      "image_light": "logo_smash.svg",
      "image_dark": "logo_smash_dark.svg",
    },
}

html_context = {
    "default_mode": "light"
}

html_css_files = [
    "css/smash.css",
]

html_use_modindex = True

bibtex_bibfiles = ["_static/bib/references.bib"]

bibtex_reference_style = "author_year"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

version = release
