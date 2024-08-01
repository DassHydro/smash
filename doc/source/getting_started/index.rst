.. _getting_started:

===============
Getting Started
===============

------------
Installation
------------

`smash` can be installed with pip on **Linux**, **Windows** or from source and supports Python versions **3.9** to **3.11**.

.. note::

    We strongly recommend using `smash` on **Linux**, particularly if you are using it on a large dataset, as 
    Fortran parallel computation is not supported on **Windows**. 

If you already have Python, you can install `smash` with:

.. code-block:: none

    pip install hydro-smash

.. note::
    
    If you have any incompatibility issue with your system, we recommand using a virtual environment such as `Anaconda <https://www.anaconda.com/>`__.

    To install `smash` follow the commands:

    .. code-block:: none

        conda create -n smash python=3.11 
        conda activate smash
        pip install hydro-smash

------
Import
------

To access `smash` and its functions import it in your Python code like this:

.. code-block:: python

    import smash

Because of a name conflict on `PyPI <https://pypi.org/>`__, the distribution name (i.e. the name used in pip
install, ``hydro-smash``) is not the same as the package name (i.e. the name used to import, ``smash``).

---------
Tutorials
---------

For a brief user guide to get started with `smash`, you can refer to the ``Quickstart`` section in the :ref:`User Guide <user_guide>`. Other in-depth functionalities and advanced optimization techniques are also included in this :ref:`User Guide <user_guide>`.

For detailed descriptions of the `smash` API, you can visit the :ref:`API Reference <api_reference>` section.

Additionally, if you're interested in the mathematical and numerical documentation of the hydrological model
operators and the tools for its calibration, including optimization tools, you can refer to the
:ref:`Math/Num Documentation <math_num_documentation>` section.
