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
import inspect
import os
import pathlib
import sys
import warnings

import smash

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("./_ext"))


# -- Project information -----------------------------------------------------

project = "smash"
copyright = "2022-2024, INRAE"
author = "INRAE"

# The full version, including alpha/beta/rc tags
release = smash.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinxcontrib.bibtex",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.linkcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_design",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autosummary_accessors",
    "custom_options_directive",
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
exclude_patterns = ["release/template-notes.rst"]

default_role = "autolink"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_favicon = "_static/favicon.ico"

html_theme = "pydata_sphinx_theme"

html_last_updated_fmt = "%b %d, %Y"

html_theme_options = {
    "github_url": "https://github.com/DassHydro/smash",
    "collapse_navigation": True,
    "logo": {
        "image_light": "logo_smash.svg",
        "image_dark": "logo_smash_dark.svg",
    },
    "navbar_start": ["corporate-logo", "navbar-logo"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
}

html_context = {"default_mode": "light"}

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

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}


# https://stackoverflow.com/questions/26134026/how-to-get-the-current-checked-out-git-branch-name-through-pygit2
def get_active_branch_name():
    head_dir = pathlib.Path("../.git/HEAD")
    with head_dir.open("r") as f:
        content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


# based on numpy doc/source/conf.py
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            with warnings.catch_warnings():
                # Accessing deprecated objects will generate noisy warnings
                warnings.simplefilter("ignore", FutureWarning)
                obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(smash.__file__))

    branch = get_active_branch_name()

    if "+" in smash.__version__:
        return f"https://github.com/DassHydro/smash/blob/{branch}/smash/{fn}{linespec}"
    else:
        return f"https://github.com/DassHydro/smash/blob/v{smash.__version__}/smash/{fn}{linespec}"
