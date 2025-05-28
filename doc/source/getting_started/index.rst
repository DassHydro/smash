.. _getting_started:

===============
Getting Started
===============

------------
Installation
------------

`smash` can be installed with pip on **Linux**, **macOS**, **Windows** or from source and supports Python
versions |min_py_version| to |max_py_version|.

.. note::

    We strongly recommend using `smash` on **Linux** or **macOS**, especially when working with large datasets, as Fortran parallel computation is not supported on **Windows**.

If you already have Python, you can install `smash` with:

.. code-block:: none

    pip install hydro-smash

.. note::

    If you encounter any compatibility issues with your system, we recommend using a virtual environment such
    as `Anaconda <https://www.anaconda.com/>`__.

    To install `smash`, use the following commands:

    .. code-block:: none

        conda create -n smash python
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
