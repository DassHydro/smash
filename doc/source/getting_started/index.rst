.. _getting_started:

===============
Getting Started
===============

------------
Installation
------------

`smash` can be installed via `pip` on **Linux**, **macOS**, or **Windows**, and supports Python versions |min_py_version| to |max_py_version|. 
It can also be installed directly from its source repository on GitHub.

.. note::

    We strongly recommend using `smash` on **Linux** or **macOS**, especially when working with large datasets, as Fortran parallel computation is not supported on **Windows**.

To install the latest released version, run:

.. code-block:: none

    pip install hydro-smash

To install the latest development version directly from the current main branch of the GitHub repository, use:

.. code-block:: none

    pip install git+https://github.com/DassHydro/smash.git

.. note::

    If you encounter any compatibility issues with your system, we recommend using a virtual environment such
    as `Anaconda <https://www.anaconda.com/>`__.

    To install `smash` with the highest supported Python version, run the following commands:

    .. parsed-literal::

        conda create -n smash python=\ |max_py_version|
        conda activate smash
        pip install hydro-smash

------
Import
------

To use `smash` in your Python code, import it as follows:

.. code-block:: python

    import smash

Note that the package name for importing (``smash``) differs from the distribution name used for installation (``hydro-smash``) due to a naming conflict on `PyPI <https://pypi.org/>`__.

---------
Tutorials
---------

To quickly get started with `smash`, refer to the ``Quickstart`` section in the :ref:`User Guide <user_guide>`.
This :ref:`User Guide <user_guide>` also includes examples of classical uses, in-depth functionalities and advanced optimization techniques.

For comprehensive details about the `smash` API, visit the :ref:`API Reference <api_reference>` section.

If you're interested in the mathematical and numerical foundations of the hydrological model operators, as well as tools for calibration and optimization, see the :ref:`Math/Num Documentation <math_num_documentation>` section.
