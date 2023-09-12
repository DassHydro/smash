.. _developers_guide:

================
Developers Guide
================

Are you a new `smash` contributor? Here are the instructions to complete your development for smash.

Installation
************

Clone the Git repository:

.. code-block:: none

    git clone https://github.com/DassHydro-dev/smash.git

If you prefer working on the Git repository without authentication required, you can create a personal access token by following these `detailed instructions <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token>`__.
Once you have generated a token, clone the Git repository using the following command:

.. code-block:: none

    git clone https://username:token@github.com/DassHydro-dev/smash.git

Replace ``username`` with your GitHub username and ``token`` with the personal access token you generated.

Go to the smash repository:

.. code-block:: none

    cd smash

It is recommended that you install the smash development enviroment using `Anaconda <https://www.anaconda.com/>`__:

.. code-block:: none

    conda env create -f environment-dev.yml

Activate the conda environment:

.. code-block:: none

    conda activate smash-dev

Compile the code and install the Python library:

.. code-block:: none

    (smash-dev) make

Make a unit test to verify the installation:

.. code-block:: none

    (smash-dev) make test

Coding workflow
***************

Once you have completed the installation, you can now start your development for smash.

Switch to the latest development version
----------------------------------------

Switch to the latest development version of smash (``maintenance/x.y.z``):

.. code-block:: none

    (smash-dev) git checkout maintenance/x.y.z

Create a new branch on which you will implement your development:

.. code-block:: none

    (smash-dev) git checkout -b your-development-branch

Fully recompile the code with the latest development version:

.. code-block:: none

    (smash-dev) make clean
    (smash-dev) make

Fortran coding style
--------------------

Some convention on the content of Fortran files:

- Write in lower case (even if the code is case insensitive, just for consistency).
- Use 4 spaces indentation.
- Use snake_case as multiple-word identifier format (i.e. **subroutine get_foo()**).
- Strongly prefer the use of module. This allows to use the ``use`` statement and to trace subroutine calls within the files (i.e. **use m_foo, only: get_foo**)
- Do not create global variables which are not well handled by the wrapper.
- Use single precision floating-point ``sp`` variable in ``md_constant.f90`` (i.e. **real(sp) :: foo**)

If you want to integrate a new Fortran file, a naming convention must be respected in order to make the different automatic installation processes understand
if the file is a module and if it must be wrapped and/or differentiated.

The structure of a Fortran file name can be written as follows: ``<prefix>_<name>.f90``.

There are no constraints on ``<name>`` here are those on the ``<prefix>``:

- ``m``: the file is a module (i.e. ``m_array_creation.f90``)
- ``mw``: the file is a module and is wrapped (i.e. ``mw_optimize.f90``)
- ``md``: the file is a module and is differentiated (i.e. ``md_constant.f90``)
- ``mwd``: the file is a module, is wrapped and differentiated (i.e. ``mwd_setup.f90``)

Other development commands
--------------------------

Here are some other commands that may be necessary for your development:

- Compile the adjoint and tangent linear model. This step is required if you are developing on the Fortran interface, particularly for the development on differentiated routines.

.. code-block:: none

    (smash-dev) make tap

- Compile the code in debug mode. This allows you to compile the Fortran interface with warning flags (i.e. ``-Wall``, ``-Wextra``, etc.).

.. code-block:: none

    (smash-dev) make debug

- If your changes are only made on the Python interface, you don't need to recompile all the code each time, but only the Python interface:

.. code-block:: none

    (smash-dev) make library

.. note::

    Whenever a modification is made to any Python file, it is necessary to recompile the Python interface by executing the ``make library`` command. 
    To avoid having to perform this step manually each time a change is made, the code can be automatically updated using the following command:

    .. code-block:: none

        (smash-dev) make library-edit


Peer review preparation
***********************

Upon completion of your coding development, it is imperative to ensure that all existing tests have been passed:

.. code-block:: none

    (smash-dev) make test

Unless any tests have been intentionally modified or added to accommodate new development. 
In such cases, the testing baseline must be regenerated to reflect the changes made:

.. code-block:: none

    (smash-dev) make test-baseline

It is advisable to verify all changes on the baseline by referencing the ``diff_baseline.csv`` file in the ``smash/tests/`` directory. 
Once you have ensured that all tests are successfully passed, rename the ``smash/tests/new_baseline.hdf5`` file to ``smash/tests/baseline.hdf5`` and remove the previous version.

We also recommend that you add your contribution to the release notes of the current development version and the documentation in the ``doc/source/`` directory. 
If you intend to create new reStructuredText (``.rst``) files for the documentation in ``doc/source/``, we suggest using the following command to generate the ``.rst`` file with auto-defined label:

.. code-block:: none

    (smash-dev) cd doc/source/
    (smash-dev) python gen_rst.py path-to-your-rst-file

After returning to the Git repository, compile the documentation to apply your changes:

.. code-block:: none

    (smash-dev) make doc

The initial compilation may take a while, but subsequent compilations will only require the time it takes to compile the modified files.

.. note::

    If you encounter any issues when compiling the documentation, try cleaning the ``doc/`` directory and then recompiling the documentation. 
    This can help eliminate any potential conflicts and bugs that may be causing the issue.

    .. code-block:: none

        (smash-dev) make doc-clean
        (smash-dev) make doc

Make sure that the Git repository is cleaned and the Python files are formatted before submitting your work:

.. code-block:: none

        (smash-dev) make clean
        (smash-dev) make doc-clean
        (smash-dev) cd smash
        (smash-dev) black *.py

Submission and review process
*****************************

After completing the previous steps and committing your new branch, push it to Git using the following command:

.. code-block:: none
    
    git push --set-upstream origin your-development-branch

Next, create a new pull request to the current development branch ``maintenance/x.y.z``. 
Ensure that your commits have passed the CI/CD pipelines; otherwise, you will be required to fix them before the review process begins.
So, it is highly recommended to run the pipelines locally before opening a pull request:

.. code-block:: none

        (smash-dev) make clean
        (smash-dev) make doc-clean
        (smash-dev) make tap
        (smash-dev) make
        (smash-dev) make test
        (smash-dev) make doc

Reviewers, including other developers or relevant people, will be assigned to review your work.
At this stage, friendly discussions may occur to enhance your implementation and maintain consistency in coding style. 
The duration of this procedure may vary depending on the nature of your contribution, such as introducing new features, addressing major bug fixes, or replacing functionalities. 
Once your pull request is approved by at least one reviewer, your branch will be merged into the latest development version.
