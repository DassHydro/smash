.. _contributor_guide.development_process_details:

===========================
Development Process Details
===========================

Build from source
-----------------

.. note::

    The build process of `smash` is similar to `SciPy <https://docs.scipy.org/doc/scipy/index.html>`__ but way simpler. 
    This is why most of the information written about the `smash` build system can also be found in the 
    `Building from source <https://docs.scipy.org/doc/scipy/building/index.html>`__ section of SciPy.

System-level dependencies
*************************

`smash` uses compiled code for speed, which means you need compilers and some other system-level
(i.e, non-Python / non-PyPI) dependencies to build it on your system.

Anaconda (recommended)
''''''''''''''''''''''

If you are using `Conda <https://www.anaconda.com/>`__, all the dependencies will be installed
automatically by running the following command:

.. code-block:: none
    
    conda env create -f environment-dev.yml

It will create a Conda environment called ``smash-dev`` that can be activated as follows:

.. code-block:: none

    conda activate smash-dev

Once ``smash-dev`` is created, it is recommended to update pip and your Conda environment
to ensure you have the latest package versions and to prevent any conflicts:

.. code-block:: none

    (smash-dev) python3 -m pip install --upgrade pip
    (smash-dev) conda update --all

Linux
'''''

If you want to use the system Python and pip, you will need:

- C and Fortran compilers (typically ``gcc`` and ``gfortran``).

- Python header files (typically a package named ``python3-dev`` or ``python3-devel``)

- Java Runtime Environment (typically ``openjdk-17-jdk``)

This can be installed on **Ubuntu/Debian Linux**:

.. code-block:: none

    sudo apt-get install -y build-essential gfortran python3-pip python3-dev openjdk-17-jdk

.. note::

    - ``build-essential`` is used to install ``gcc`` and ``make``. ``make`` is optionnal but a ``Makefile`` is
      available to build `smash`, generate the documentation, generate the adjoint code, run the tests and
      format the source code.

    - ``openjdk-17-jdk`` is used to generate the adjoint with `Tapenade <https://team.inria.fr/ecuador/en/tapenade/>`__

Windows
'''''''

.. warning::
    
    Section in development. Information on compiling `smash` under **Windows** can be found in the
    ``pyproject.toml``, ``meson.build`` and ``wheel.yml`` workflow.

Build
*****

Once the system-level dependencies installed and the git cloned, `smash` can be built using either the ``make`` or ``make edit`` command.

- ``make``
    This command build `smash` and install it in the current environment. This command is equivalent to:

    .. code-block:: none

        pip install .

    .. hint::

        Remembering that we need to move out of the root of the repository to ensure we pick up the package
        and not the local ``smash/`` source directory.

        .. code-block:: none

            cd doc
            python -c "import smash; print(smash.__version__)"

- ``make-edit``
    This command build `smash` and install it in editable mode. It automatically rebuilds itself if a file is
    modified, whether in Python or Fortran. This command is equivalent to:

    .. code-block:: none

        pip install --no-build-isolation --config-settings=editable-verbose=true --editable .

.. warning::
    The adjoint code is not part of the build but is directly a source of `smash`. If you modify a file
    that needs to be differentiated, you will still need to regenerate the adjoint code. See the section
    :ref:`contributor_guide.development_process_details.build_from_source.automatic_differentiation`

On **Linux**, `smash` can be built with or without dependency on `OpenMP <https://www.openmp.org/>`__.
By default, the dependency is activated. To build without it, simply add the following option to the build command:

.. code-block:: none

    make use-openmp=false
    # Equivalent to
    pip install -Csetup-args=-Duse-openmp=false .

Or in editable mode
    
.. code-block:: none

    make edit use-openmp=false
    # Equivalent to
    pip install -Csetup-args=-Duse-openmp=false --no-build-isolation --config-settings=editable-verbose=true --editable .

Understanding the build steps
*****************************

Building `smash` relies on the following tools, which can be considered part of the build system:

- ``meson``: the Meson build system, installable as a pure Python package from PyPI or conda-forge.

- ``ninja``: the build tool invoked by Meson to do the actual building (e.g. invoking compilers). 
  Installable also from PyPI (on all common platforms) or conda-forge.

- ``meson-python``: the Python build backend (i.e., the thing that gets invoked via a hook in ``pyproject.toml``
  by a build frontend like pip or pypa/build). This is a thin layer on top of Meson, with as main roles (a)
  interface with build frontends, and (b) produce sdists and wheels with valid file names and metadata.

.. hint::

    More information can be found for each tool here:

    - `meson <https://mesonbuild.com/>`__ 
    - `ninja <https://ninja-build.org/>`__ 
    - `meson-python <https://meson-python.readthedocs.io/en/latest/>`__

To build `smash` under Meson, several ``meson.build`` files are used in different places in the source code.

- ``meson.build``
    This file is the root file where dependencies are managed. Python installation, Fortran and C
    compilers/flags and `F2PY <https://numpy.org/doc/stable/f2py/>`__, `OpenMP <https://www.openmp.org/>`__
    dependencies etc.

- ``smash/meson.build``
    Second file in the tree, where simply all the Python files are installed, looping through the subfolders.

- ``smash/factory/mesh/meson.build``
    Specific file managing the Python extension ``_libmesh.cpython*.so`` for the mesh. The generation of this
    Python extension consists of a call to `F2PY <https://numpy.org/doc/stable/f2py/>`__ on the
    Fortran file ``mw_mesh.f90``.

- ``smash/fcore/meson.build``
    Specific file managing the Python extension ``_libfcore.cpython*.so`` for the Fortran core. This file is
    the most complicated of all, and can be summarised as follows:

    - Declare all the sources
        First, the Python, C and Fortran sources are declared. There is no need to worry about the order in
        which Fortran files are declared. Meson takes care of managing dependencies between modules.
        However, it is important to declare the ``f90`` files in the correct variables.
        There are 5 types of file:

        - Module (i.e. file name starting with ``m_``)
            Fortran module

            .. code-block:: meson

                m_f90_sources = [
                    'common/m_array_creation.f90',
                    ...
                ]

        - Wrapped module (i.e. file name starting with ``mw_``)
            Fortran module wrapped with `f90wrap <https://github.com/jameskermode/f90wrap>`__

            .. code-block:: meson
           
                mw_f90_sources = [
                     'forward/mw_forward.f90',
                    ...
                ]

        - Differentiated module (i.e. file name starting with ``md_``)
            Fortran module differentiated with `Tapenade <https://team.inria.fr/ecuador/en/tapenade/>`__

            .. code-block:: meson
           
                md_f90_sources = [
                    'common/md_constant.f90',
                    ...
                ]

        - Wrapped and differentiated module (i.e. file name starting with ``mwd_``)
            Fortran module wrapped and differentiated with `f90wrap <https://github.com/jameskermode/f90wrap>`__
            and `Tapenade <https://team.inria.fr/ecuador/en/tapenade/>`__

            .. code-block:: meson
           
                mwd_f90_sources = [
                    'derived_type/mwd_atmos_data.f90',
                    ...
                ]

        - Non module file (i.e. file name starting without prefix)
            Simple Fortran file.

            .. code-block:: meson
           
                f90_sources = [
                    'forward/forward.f90',
                ] + m_f90_sources + md_f90_sources + mw_f90_sources + mwd_f90_sources

            .. note::

                We recommend that you do not insert a Fortran file without a module, for reasons of wrapping
                and readability. The only file of this type contains the **top differentiation routine**.

    - Choose adjoint file
        Two adjoint files are available but only one must be used depending on the OS and ``use-openmp``
        option. On **Windows**, only choose the non `OpenMP <https://www.openmp.org/>`__ file.

        .. code-block:: meson

            if host_machine.system() == 'windows'
                f90_sources += 'forward/forward_db.f90'
            else
                if get_option('use-openmp')
                    f90_sources += 'forward/forward_openmp_db.f90'
                else
                    f90_sources += 'forward/forward_db.f90'
                endif
            endif

        .. note::

            The ``use-openmp`` option is declared in the ``meson.options`` file

    - Generate f90wrap files
        Once all the sources declared, we can call `f90wrap <https://github.com/jameskermode/f90wrap>`__ to
        generate the Python and Fortran wrapped files. Generated Python files will be installed and Fortran
        files used to generate the Python extension. To generate this files, a call to a self-made 
        ``f90wrap/generate_f90wrap.py`` file is done. It is just a wrapper around the ``f90wrap`` command
        to handle monkey patchings and build directory.

        .. code-block:: meson

            f90wrap = [
                py,
                files('../../f90wrap/generate_f90wrap.py'),
                '@INPUT@',
                '-k', files('../../f90wrap/kind_map'),
                '--build-dir', '@OUTDIR@',
            ]
            ...
            f90wrap_sources = custom_target(
                input: mw_f90_sources + mwd_f90_sources,
                output: [f90wrap_f90_output, f90wrap_py_output],
                command: f90wrap + ['-m', 'libfcore'],
                install: true,
                install_dir: [f90wrap_f90_install_dir, f90wrap_py_install_dir],
            )
    
    - Generate F2PY files
        Once the Fortran `f90wrap <https://github.com/jameskermode/f90wrap>`__ files are generated, we can
        call `F2PY <https://numpy.org/doc/stable/f2py/>`__ to generate the C file used generate the
        Python extension.

        .. code-block:: meson

            f2py_f90wrap = ['f2py-f90wrap', '@INPUT@', '--build-dir', '@OUTDIR@', '--lower']

            f2py_f90wrap_sources = custom_target(
                input: f90wrap_f90_sources,
                output: ['_libfcoremodule.c', '_libfcore-f2pywrappers.f'],
                command: f2py_f90wrap + ['-m', '_libfcore'],
            )

    - Handle dependencies and link arguments
        Juste before generating the Python extension, dependencies and link arguments are declared depending
        on the OS and ``use-openmp``. On **Windows**, ``libquadmath`` must be explicitly linked and we link
        all the libraries staticly with ``-static``.

        .. code-block:: meson

            link_args = []
            dependencies = [fortranobject_dep]
            if host_machine.system() == 'windows'
                link_args += ['-lquadmath', '-static']
            else
                if get_option('use-openmp')
                    dependencies += openmp_dep
                endif
            endif

Fortran guideline
-----------------

Global convention
*****************

The aim of this section is to show how to integrate new functions into the Fortran code of smash.

Style convention
''''''''''''''''

Here are the conventions that have been applied on the content of a Fortran file (most of the time ...):

- Use lowercase for all Fortran constructs (``do``, ``subroutine``, ``module``, ...)

- For other names use all lowercase and ``snake_case`` as multiple-word identifier format (``optimize``, ``get_parameters``, ``set_states``, ...).

- Use 4 spaces indentation.

.. note::
    `fprettify <https://github.com/pseewald/fprettify>`__ is used to format Fortran file. It can be used as follows:

    .. code-block:: none

        fprettify --indent 4 mwd_parameters.f90
        fprettify --indent 4 *.f90

    or using the make ``format`` command

    .. code-block:: none

        make format

.. _contributor_guide.development_process_details.fortran_guideline.global_convention.file_name_convention:

File name convention
''''''''''''''''''''

If you want to integrate a new Fortran file, a naming convention must be respected in order to make the different installation 
processes understand if the file is a module and if it must be wrapped and/or differentiated.

The structure of a Fortran file name can be written as follows: ``<prefix>_<name>.f90`` using lowercase and ``snake_case`` 
as multiple-word identifier format.

There are no constraints on ``<name>`` here are those on the ``<prefix>``:

- ``m``: the file is a module (``m_array_creation.f90``)

- ``mw``: the file is a module and is wrapped (``mw_optimize.f90``)

- ``md``: the file is a module and is differentiated (``md_constant.f90``)

- ``mwd``: the file is a module, is wrapped and differentiated (``mwd_setup.f90``)

.. note::
    We strongly recommand the use of module. Specifically if the file contains sources to be wrapped or differentiated.

.. _contributor_guide.development_process_details.fortran_guideline.global_convention.floating_point_convention:

Floating point convention
'''''''''''''''''''''''''

Most of the real variables are single precision floating-point. In some functions, these variables are casted into double precision floating-point.
Therefore, two constants ``sp`` and ``dp`` are used to precise the floating-point precision, respectively, simple precision and double precision.

.. code-block:: fortran

    real(sp) :: foo = 2._sp 
    real(dp) :: bar = 0._dp

    bar = real(foo, dp)

Compile
*******

Compile a pre-existing file
'''''''''''''''''''''''''''

If you are editing a pre-existing file, there are no particular constraints before compiling the code.
Compile with the following command:

.. code-block:: none

    make

.. note::

    If you have already built `smash` in editable mode (i.e. ``make edit``). You do not need to call the ``make``
    or ``make edit`` command. You can directly execute your Python script, `smash` will be automatically
    rebuilded

.. _contributor_guide.development_process_details.fortran_guideline.compile.compile_a_new_file:

Compile a new file
''''''''''''''''''

If you are creating a new file, respecting the naming convention 
(:ref:`contributor_guide.development_process_details.fortran_guideline.global_convention.file_name_convention`),
you must fill in the ``samsh/fcore/meson.build`` file to declare the new file

Wrapping
********

The Fortran code is wrapped using the `f90wrap <https://github.com/jameskermode/f90wrap>`__ library. Here are the different steps to wrap `smash` 
code efficiently. We assume here that we are integrating a wrapped module from scratch. Certain steps can be repeated if you are adding to
pre-existing files.

.. hint::
    Quite a few examples are also available in the f90wrap GitHub directory in the examples folder 
    (see `here <https://github.com/jameskermode/f90wraptree/master/examples>`__

.. _contributor_guide.development_process_details.fortran_guideline.wrapping.vector2_case:

Vector2 case
''''''''''''

We are going to create a derived type called ``Vector2DT`` containing two real variables, ``x`` and ``y``, and a set of subroutines/functions 
associated with this derived type.

Create new wrapped files
""""""""""""""""""""""""

As explained in the :ref:`contributor_guide.development_process_details.fortran_guideline.global_convention.file_name_convention` section, a 
Fortran file will be automatically wrapped if it name contains the prefix ``mw`` or ``mwd``. We will consider the following 
Fortran files: ``mw_vector2.f90`` and ``mw_vector2_manipulation.f90``. The first file will contain the implementation of the derived type 
``Vector2DT`` and the second will contain all the subroutines/functions that manipulate the derived type. It might well have been possible to 
do everything in a single file, but it was decided in `smash` to separate them. 

- ``mw_vector2.f90`` (this file can be stored in the folder ``smash/fcore/derived_type``)

.. code-block:: fortran

    module mw_vector2
    ...
    end module mw_vector2

- ``mw_vector2_manipulation.f90`` (this file can be stored in the folder ``smash/fcore/routine``)
  
.. code-block:: fortran

    module mw_vector2_manipulation
    ...
    end module mw_vector2_manipulation

.. note::
    The entire file will be wrapped, so it is advisable to separate the functions to be wrapped from those that are not.

The files (even empty ones) can be compiled and wrapped (see the :ref:`contributor_guide.development_process_details.fortran_guideline.compile.compile_a_new_file` section) 
and imported in Python as follows:

.. code-block:: python

    >>> import smash.fcore._mw_vector2
    >>> import smash.fcore._mw_vector2_manipulation

Derived type implementation
"""""""""""""""""""""""""""

First, we will implement the derived type ``Vector2DT`` in the ``mw_vector2.f90`` file.

.. note::
    We add the suffix ``DT`` for each derived type because Fortran is case insensitive and will not differentiate between ``vector2`` 
    and ``Vector2``.

.. code-block:: fortran

    module mw_vector2

        use md_constant, only: sp

        implicit none

        type Vector2DT

            real(sp) :: x
            real(sp) :: y

        end type Vector2DT

    end module mw_vector2

.. note::
    ``sp`` is equal to ``4``, it is simple precision 
    (see the :ref:`contributor_guide.development_process_details.fortran_guideline.global_convention.floating_point_convention` section)

A wrapped derived type is interpreted as a Python class. Let's compile, initialize it and view what it contains:

.. code-block::

    >>> from smash.fcore._mw_vector2 import Vector2DT
    >>> v = Vector2DT()
    >>> v
    Vector2DT
        x: 4.201793856028541e+18
        y: 3.0741685710357837e-41
    >>> v.x
    4.201793856028541e+18
    >>> v.y
    3.0741685710357837e-41

We can see that the 2 variables, ``x`` and ``y`` present in the original derived type are accessible in Python as class properties but filled with garbage values because they were not
initialized. There two ways to initialize the values of a derived type:

- Assign values in the declaration of the derived type variables

.. code-block:: fortran

    module mw_vector2

        use md_constant, only: sp

        implicit none

        type Vector2DT

            real(sp) :: x = 0._sp
            real(sp) :: y = 0._sp

        end type Vector2DT

    end module mw_vector2

- Create a specific initialization subroutine which will be interpreted as a Python class constructor (``__init__`` function). 
  `f90wrap <https://github.com/jameskermode/f90wrap>`__ will automatically detects derived type initialization subroutine
  if the subroutine name follows the convention: ``<derived-type-name>_initialise``. In our case, the subroutine must be called:
  ``Vector2DT_initialise``. Let's write the initialization subroutine after adding the ``contains`` statement.

.. code-block:: fortran

    module mw_vector2

        use md_constant, only: sp

        implicit none

        type Vector2DT

            real(sp) :: x
            real(sp) :: y

        end type Vector2DT

    contains

        subroutine Vector2DT_initialise(this)

            implicit none

            type(Vector2DT), intent(inout) :: this

            this%x = 0._sp
            this%y = 0._sp

        end subroutine Vector2DT_initialise

    end module mw_vector2

The two methods in this example are equivalent and here is the result in Python:

.. code-block:: python

    >>> v = Vector2DT()
    >>> v
    Vector2DT
        x: 0.0
        y: 0.0

We successfully initialize the derived type with default values. However, the second method, using an initialization function, is more flexible. 
We can, for example, not define default values but initialize the derived type with values from Python. Let's rewrite the initialize subroutine and
add arguments.

.. code-block:: fortran

    module mw_vector2

        use md_constant, only: sp

        implicit none

        type Vector2DT

            real(sp) :: x
            real(sp) :: y

        end type Vector2DT

    contains

        subroutine Vector2DT_initialise(this, x, y)

            implicit none

            type(FooDT), intent(inout) :: this
            real(sp), intent(in) :: x
            real(sp), intent(in) :: y

            this%x = x
            this%y = y

        end subroutine Vector2DT_initialise

    end module mw_vector2

We add 2 arguments which correspond to each variable of the derived type to initialize. On the Python side, this is how it translates:

.. code-block:: python

    >>> v = Vector2DT(0, 0)
    >>> v
    Vector2DT
        x: 0.0
        y: 0.0
    >>> v = Vector2DT(1, 1)
    >>> v
    Vector2DT
        x: 1.0
        y: 1.0

It is also possible to modify the values once initialization is complete, since each element of the derived type is a property with a getter and a setter.

.. code-block:: python

    >>> v = Vector2DT(0, 0)
    >>> v
    Vector2DT
        x: 0.0
        y: 0.0
    >>> v.x = 3
    >>> v.y = 2
    v
    Vector2DT
        x: 3.0
        y: 2.0

Functions implementation
""""""""""""""""""""""""

We can now implement a number of subroutines/functions in the ``mw_vector2_manipulation.f90`` file to manipulate this derived type. We need first to import the module where
the ``Vector2DT`` derived type is defined ``mw_vector2`` and them in a ``contains`` statement add the functions.

.. code-block:: fortran

    module mw_vector2_manipulation

        use md_constant, only: sp
        use mw_vector2, only: Vector2DT

        implicit none

    contains

        function vector2_add_value(v, add) result(res)

            type(Vector2DT), intent(in) :: v
            real(sp), intent(in) :: add

            type(Vector2DT) :: res

            res%x = v%x + add
            res%y = v%y + add

        end function vector2_add_value

        function vector2_dot_product(v1, v2) result(res)

            type(Vector2DT), intent(in) :: v1
            type(Vector2DT), intent(in) :: v2

            real(sp) :: res

            res = v1%x*v2%x + v1%y*v2%y

        end function vector2_dot_product

    end module mw_vector2_manipulation

We have added two functions, one to add a value to each element of the ``Vector2DT`` and the other one to compute 
the dot product between two ``Vector2DT`` , so let's see how this translates into Python:

.. code-block:: python

    >>> from smash.fcore._mw_vector2 import Vector2DT
    >>> from smash.fcore._mw_vector2_manipulation import vector2_add_value, vector2_dot_product
    
    >>> v = Vector2DT(0, 0)
    >>> vector2_add_value(v, 5)
    Vector2DT
        x: 5.0
        y: 5.0

    >>> v1 = Vector2DT(1, 1)
    >>> v2 = Vector2DT(2, 3)
    >>> vector2_dot_product(v1, v2)
    5.0

This completes the first example of Fortran wrapping in `smash`. The next examples will be less detailed but will aim to expose a wider range of functionality, 
variable types, allocation management, string management, etc.

Matrix2 case
''''''''''''

We are going to create a derived type called ``Matrix2DT`` containing one allocatable real variable of 2 dimensions, ``vle``, two integer variables
representing the number of rows and columns of the matrix, ``n`` and ``m``, respectively and a set of subroutines/functions associated with 
this derived type. Similar to the :ref:`contributor_guide.development_process_details.fortran_guideline.wrapping.vector2_case` section, two files
are created, ``mw_matrix2.f90`` and ``mw_matrix2_manipulation.f90``. The aim of this case is to illustrate how arrays can are handled.

- ``mw_matrix2.f90`` (this file can be stored in the folder ``smash/fcore/derived_type``)

.. code-block:: fortran

    module mw_matrix2
    
        use md_constant, only: sp

        implicit none

        type Matrix2DT

            integer :: n
            integer :: m
            real(sp), dimension(:, :), allocatable :: vle

        end type Matrix2DT
    
    contains

        subroutine Matrix2DT_initialise(this, n, m, vle0)

            implicit none

            type(Matrix2DT), intent(inout) :: this
            integer, intent(in) :: n, m
            real(sp), intent(in) :: vle0
            
            this%n = n
            this%m = m
            allocate (this%vle(this%n, this%m))
            this%vle(:, :) = vle0

        end subroutine Matrix2DT_initialise

    end module mw_matrix2

- ``mw_matrix2_manipulation.f90`` (this file can be stored in the folder ``smash/fcore/routine``)

.. code-block:: fortran

    module mw_matrix2_manipulation
    
        use md_constant, only: sp
        use mw_matrix2, only: Matrix2DT, Matrix2DT_initialise

        implicit none
    
    contains

        function matrix2_add_value(mat, add) result(res)

            implicit none

            type(Matrix2DT), intent(inout) :: mat
            real(sp), intent(in) :: add

            type(Matrix2DT) :: res

            call Matrix2DT_initialise(res, mat%n, mat%m, 0._sp)

            res%vle(:, :) = mat%vle(:, :) + add

        end function matrix2_add_value

        function matrix2_transpose(mat) result(res)

            implicit none

            type(Matrix2DT), intent(in) :: mat
            
            type(Matrix2DT) :: res
            integer :: i, j

            call Matrix2DT_initialise(res, mat%m, mat%n, 0._sp)

            ! Could also use directly the Fortran intrinsic function TRANSPOSE
            ! res%vle = TRANSPOSE(mat%vle)
            do i = 1, mat%m
                do j = 1, mat%n
                    res%vle(i, j) = mat%vle(j, i)
                end do
            end do

        end function matrix2_transpose

    end module mw_matrix2_manipulation

This translates into Python:

.. code-block:: python

    >>> from smash.fcore._mw_matrix2 import Matrix2DT

    >>> mat = Matrix2DT(2, 3, 0)
    >>> mat
    Matrix2DT
        m: 3
        n: 2
        vle: array([[0., 0., 0.],
                    [0., 0., 0.]], dtype=float32)
    >>> type(mat.vle)
    <class 'numpy.ndarray'>

Fortran arrays are casted to `numpy.ndarray` when accessed in Python. So all the methods associated with a `numpy.ndarray` can be used.

.. code-block:: python

    >>> from smash.fcore._mw_matrix2 import Matrix2DT
    >>> from smash.fcore._mw_matrix2_manipulation import matrix2_add_value, matrix2_transpose

    >>> mat = Matrix2DT(2, 3, 0)
    >>> mat
    Matrix2DT
        m: 3
        n: 2
        vle: array([[0., 0., 0.],
                    [0., 0., 0.]], dtype=float32)

    >>> mat.vle.shape
    (2, 3)
    >>> mat.vle.dtype
    dtype('float32'

    >>> mat.vle[0, :] = 2
    >>> mat
        Matrix2DT
        m: 3
        n: 2
        vle: array([[2., 2., 2.],
                    [0., 0., 0.]], dtype=float32)

    >>> matrix2_add_value(mat, 4)
    Matrix2DT
        m: 3
        n: 2
        vle: array([[6., 6., 6.],
                    [4., 4., 4.]], dtype=float32)

    >>> matrix2_transpose(mat)
    Matrix2DT
        m: 2
        n: 3
        vle: array([[0., 2.],
                    [0., 2.],
                    [0., 2.]], dtype=float32)

Matrix2Array case
'''''''''''''''''

We are going to create a derived type called ``Matrix2ArrayDT`` containing one allocatable ``Matrix2DT`` type variable of 1 dimension.
The aim of this case is to illustrate how derived type arrays are handled. We will keep the previous files created for ``Matrix2DT``
(i.e. ``mw_matrix2.f90`` and ``mw_matrix2_manipulation.f90``).

.. code-block:: fortran

    module mw_matrix2
    
        use md_constant, only: sp

        implicit none

        type Matrix2DT

            integer :: n
            integer :: m
            real(sp), dimension(:, :), allocatable :: vle

        end type Matrix2DT

        type Matrix2ArrayDT

            integer :: n
            type(Matrix2DT), dimension(:), allocatable :: mat

        end type Matrix2ArrayDT
    
    contains

        subroutine Matrix2DT_initialise(this, n, m, vle0)

            implicit none

            type(Matrix2DT), intent(inout) :: this
            integer, intent(in) :: n, m
            real(sp), intent(in) :: vle0
            
            this%n = n
            this%m = m
            allocate (this%vle(this%n, this%m))
            this%vle(:, :) = vle0

        end subroutine Matrix2DT_initialise

        subroutine Matrix2ArrayDT_initialise(this, n, n_arr, m_arr, vle0_arr)

            implicit none

            type(Matrix2ArrayDT), intent(inout) :: this
            integer, intent(in) :: n
            integer, dimension(n), intent(in) :: n_arr, m_arr
            real(sp), dimension(n), intent(in) :: vle0_arr
            
            integer :: i

            this%n = n

            allocate (this%mat(this%n))

            do i = 1, this%n

                call Matrix2DT_initialise(this%mat(i), n_arr(i), m_arr(i), vle0_arr(i))

            end do

        end subroutine Matrix2ArrayDT_initialise

    end module mw_matrix2

Here we create a derived type ``Matrix2ArrayDT`` which contains an array of ``Matrix2DT``. To initialize this derived type, we pass the number of
``Matrix2DT`` that we want to allocate ``n``, the number of rows and columns for each allocated matrix ``n_arr`` and ``m_arr``, respectively and 
an initial value for each ``vle0_arr``. This translates into Python:

.. code-block:: python

    >>> import numpy as np
    >>> from smash.fcore._mw_matrix2 import Matrix2ArrayDT

    >>> n = 2
    >>> n_arr = np.array([2, 3], dtype=np.int32)
    >>> m_arr = np.array([4, 1], dtype=np.int32)
    >>> vle_arr = np.array([1, 5], dtype=np.float32)
 
    >>> mat_arr = Matrix2ArrayDT(n, n_arr, m_arr, vle_arr)
    >>> mat_arr
    Matrix2ArrayDT
        mat: ['Matrix2DT', 'Matrix2DT']
        n: 2

It allows us to create an array of ``Matrix2DT`` that can have different shapes. Here ``(2, 4)`` and ``(3, 1)``. We can iterate over as follows:

.. code-block:: python

    >>> import numpy as np
    >>> from smash.fcore._mw_matrix2 import Matrix2ArrayDT

    >>> n = 2
    >>> n_arr = np.array([2, 3], dtype=np.int32)
    >>> m_arr = np.array([4, 1], dtype=np.int32)
    >>> vle_arr = np.array([1, 5], dtype=np.float32)
 
    >>> mat_arr = Matrix2ArrayDT(n, n_arr, m_arr, vle_arr)
    >>> mat_arr
    Matrix2ArrayDT
        mat: ['Matrix2DT', 'Matrix2DT']
        n: 2

    >>> for m in mat_arr.mat.items():
    >>>     m
    Matrix2DT
        m: 4
        n: 2
        vle: array([[1., 1., 1., 1.],
                    [1., 1., 1., 1.]], dtype=float32)
    Matrix2DT
        m: 1
        n: 3
        vle: array([[5.],
                    [5.],
                    [5.]], dtype=float32)
    
    >>> mat_arr.mat[0]
    Matrix2DT
        m: 4
        n: 2
        vle: array([[1., 1., 1., 1.],
                    [1., 1., 1., 1.]], dtype=float32)

    >>> mat_arr.mat[1]
    Matrix2DT
        m: 1
        n: 3
        vle: array([[5.],
                    [5.],
                    [5.]], dtype=float32)

Character/String case
'''''''''''''''''''''

We are going to create a derived type called ``CharacterDT`` containing a character ``c`` and character array ``c_arr`` in order to get 
into the details of this specific edge case of the wrapping and how we handle it in `smash`. 
Let's create a ``mw_character.f90`` file.

.. code-block:: fortran

    module mw_character

        use md_constant, only: sp, lchar

        implicit none

        type CharacterDT

            character(lchar) :: c = "foo"
            character(lchar), dimension(2) :: c_arr = "bar"

        end type CharacterDT

    end module mw_character

This translates into Python:

.. code-block:: python

    >>> from smash.fcore._mw_character import CharacterDT
    >>> char = CharacterDT()
    >>> char
    CharacterDT
        c: b'foo'
        c_arr: array([[ 98,  98],
        [ 97,  97],
        [114, 114],
        ...
        [ 32,  32],
        [ 32,  32]], dtype=uint8)
    
    >>> type(char.c)
    <class 'bytes'>
    >>> type(char.c_arr), char.c_arr.dtype
    <class 'numpy.ndarray'>, dtype('uint8')

As you can see, when wrapped to Python, a Fortran character is interpreted as ``bytes`` and character array as a `numpy.ndarray` of dtype ``uint8`` 
(unsigned 8 bits integer). To get something interpretable, we can cast ``bytes`` to ``str`` with the ``decode`` method and decode each ``ASCII``
value in the character array.

.. code-block:: python

    >>> from smash.fcore._mw_character import CharacterDT
    >>> char = CharacterDT()
    >>> char
    CharacterDT
        c: b'foo'
        c_arr: array([[ 98,  98],
        [ 97,  97],
        [114, 114],
        ...
        [ 32,  32],
        [ 32,  32]], dtype=uint8)

    >>> char.c.decode()
    'foo'

    # Cast to bytes
    >>> char.c_arr.tobytes(order="F")
    b'bar
                    bar
                                         '
    # Decode with utf-8 encoding
    >>> char.c_arr.tobytes(order="F").decode()
    'bar
                    bar
                                         '
    # Split by whitespaces
    >>> char.c_arr.tobytes(order="F").decode().split()
    ['bar', 'bar']

We have managed to interpret these values, but it's not particularly conveniente. Moreover, how can we change the values in Python ?

.. code-block:: python

    >>> from smash.fcore._mw_character import CharacterDT
    >>> char = CharacterDT()
    >>> char
    CharacterDT
        c: b'foo'
        c_arr: array([[ 98,  98],
        [ 97,  97],
        [114, 114],
        ...
        [ 32,  32],
        [ 32,  32]], dtype=uint8)

    >>> char.c = "baz"
    >>> char.c
    "baz"
    >>> char.c_arr = ["buz", "buz"]
    ValueError: invalid literal for int() with base 10: 'buz'
    >>> char.c_arr = np.array(["buz", "buz"])
    ValueError: invalid literal for int() with base 10: 'buz'

It's ok for a character but not for the character array. To get around this, some self made Fortran directives can be inserted at the definition
of the variables in the derived type. ``!$F90W char`` for character and ``!$F90W char-array`` for character array.

.. code-block:: fortran

    module mw_character

        use md_constant, only: sp, lchar

        implicit none

        type CharacterDT

            character(lchar) :: c = "foo" !$F90W char
            character(lchar), dimension(2) :: c_arr = "bar" !$F90W char-array

        end type CharacterDT

    end module mw_character

This allows us to manipulate this derived type in Python in a more practical way:

.. code-block:: python

    >>> import numpy as np
    >>> from smash.fcore._mw_character import CharacterDT
    >>> char = CharacterDT()
    >>> char
    CharacterDT
        c: 'foo'
        c_arr: array(['bar', 'bar'], dtype='<U3')
    
    >>> char.c = "baz"
    >>> char.c_arr = np.array(["buz", "buz"])
    >>> char
    CharacterDT
        c: 'baz'
        c_arr: array(['buz', 'buz'], dtype='<U3')

How it works? The file is parsed and for each directive encountered, a decorator is added to the getters and setters of the f90wrap Python file 
associated. Decorators are defined in this file ``smash/fcore/_f90wrap_decorator.py``.

Array indexing case
'''''''''''''''''''

An other edge case is to manipulate values that contain indices (i.e. location of the maximum value of a matrix). Why this is a edge case ?
because Python is 0-based indexed and Fortran is 1-based indexed (by default). We will create a derived type called ``ArrayIndexDT``
containing an array of real ``r_arr`` and an integer ``ind`` in order to get into the details of this specific edge case of the wrapping 
and how we handle it in `smash`.  Let's create ``mw_array_index.f90`` and ``mw_array_index_manipulation.f90`` files.

.. code-block:: fortran

    module mw_array_index

        use md_constant, only: sp

        implicit none

        type ArrayIndexDT

            integer :: ind = 1
            real(sp), dimension(10) :: r_arr = 0._sp

        end type ArrayIndexDT

    end module mw_array_index

.. code-block:: fortran

    module mw_array_index_manipulation

        use md_constant, only: sp
        use mw_array_index, only: ArrayIndexDT

        implicit none

    contains

        function array_index_at_ind(a) result(res)

            implicit none

            type(ArrayIndexDT), intent(in) :: a

            real(sp) :: res

            res = a%r_arr(a%ind)

        end function array_index_at_ind

    end module mw_array_index_manipulation

This translates in Python:

.. code-block:: python

    >>> import numpy as np
    >>> from smash.fcore._mw_array_index import ArrayIndexDT
    >>> from smash.fcore._mw_array_index_manipulation import array_index_at_ind

    >>> ai = ArrayIndexDT()
    >>> ai
    ArrayIndexDT
        ind: 1
        r_arr: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    
    >>> ai.r_arr = np.arange(0, ai.r_arr.size)
    >>> ai
    ArrayIndexDT
        ind: 1
        r_arr: array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=float32)

Now we can for exemple store the indice of the maximum value of the array in ``ai.ind`` and try to access the maximum value back with this indice
from Python and Fortran

.. code-block:: python

    >>> ai.ind = np.argmax(ai.r_arr)
    >>> ai
    ArrayIndexDT
        ind: 9
        r_arr: array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=float32)
    
    # Access from Python
    >>> ai.r_arr[ai.ind]
    9.0

    # Access from Fortran
    >>> array_index_at_ind(ai)
    8.0

As you can see, the value are different and it's because the arrays are not indexed in the same way. The value of ``ai.ind`` is set to ``9`` which
is correct in Python but should be ``10`` in Fortran. As a result, we'd need to manipulate the index depending on whether we calculated it in 
Fortran or Python, which isn't practical and is prone to out-of-range accesses. To get around this, a self made Fortran directive ``!$F90W index`` 
can be used to substract 1 from the value of a variable storing indices when passing from Fortran to Python or add 1 the other way around.
Let's do the same thing with the new directive.

.. note::

    If the variable storing the indices is an array, the directive is ``!$F90W index-array`` instead of ``!$F90W index``.

.. code-block:: fortran

    module mw_array_index

        use md_constant, only: sp

        implicit none

        type ArrayIndexDT

            integer :: ind = 1 !$F90W index
            real(sp), dimension(10) :: r_arr = 0._sp

        end type ArrayIndexDT

    end module mw_array_index

.. code-block:: python

    >>> import numpy as np
    >>> from smash.fcore._mw_array_index import ArrayIndexDT
    >>> from smash.fcore._mw_array_index_manipulation import array_index_at_ind

    >>> ai = ArrayIndexDT()
    >>> ai
    ArrayIndexDT
        ind: 0
        r_arr: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)

    >>> ai.r_arr = np.arange(0, ai.r_arr.size)
    >>> ai
    ArrayIndexDT
        ind: 0
        r_arr: array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=float32)

    >>> ai.ind = np.argmax(ai.r_arr)
    >>> ai
    ArrayIndexDT
        ind: 9
        r_arr: array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=float32)
    
    # Access from Python
    >>> ai.r_arr[ai.ind]
    9.0

    # Access from Fortran
    >>> array_index_at_ind(ai)
    9.0

.. _contributor_guide.development_process_details.build_from_source.automatic_differentiation:

Automatic differentiation
*************************

The Fortran code is automatically differentiated using the `Tapenade <https://team.inria.fr/ecuador/en/tapenade/>`__. The use
of Tapenade can quickly become complex and a reason to give up in the development of `smash`. The aim of this
section is to explore the subtleties of this not-so-automatic differentiation. The adjoint code is not part
of the build system for reasons of managing Tapenade under different OS as well as to facilitate
debugging. Two adjoint codes are in the sources, one with `OpenMP <https://www.openmp.org/>`__ directives
``forward_openmp_db.f90`` and the other without ``forward_db.f90``. Depending on the OS and the
``use-openmp`` option, one of the two files is used.

To build the adjoint codes, a Java Runtime Environment must be installed before running the following command:

.. code-block:: none
    
    make tap

This command generates the two adjoint files.

Differentiated files
''''''''''''''''''''

A Fortran file will be automatically differentiated if it name contains the prefix ``md`` or ``mwd``.
There is just one exception with the file ``fcore/forward/forward.f90`` which is not a module and contains
the **top differentiation routine** ``base_foward_run``. This file is not a module because Tapenade is complaining about.
If an operator needs to be added to the direct model, it is necessary to implement it in a pre-existing file
containing ``md`` or ``mwd`` or to insert it in a new file containing these same prefixes. The result of the differentiation 
(i.e. the adjoint and tangent linear model) is writted in the ``fcore/forward/forward_openmp_db.f90`` and 
``fcore/forward/forward_db.f90`` file. 

.. note::

    There is no need to modify the files ``forward_openmp_db.f90`` and ``forward_db.f90``,
    apart from for debugging purposes, as this file is constantly updated with the sources as soon as the ``make tap``
    command is called.

Tapenade usage
''''''''''''''

The call to Tapenade to generate the adjoint files is made with the ``make tap`` command. This command calls 
a Python file ``tapenade/generate_tapenade.py``. The generation of the adjoints files takes place in 4 stages:

- ``Patch Fortran files``
    Before calling the Tapenade executable, this step allows you to make changes to the files to be differentiated.
    The modifications currently available are:
    
    - Delete sections of code via a pair of directives.
        For example, in the file ``fcore/forward/md_simulation.f90`` file, you can see how this pair of directives is used.

        .. code-block:: fortran

            subroutine store_time_step(setup, mesh, output, returns, checkpoint_variable, time_step)

                implicit none

                type(SetupDT), intent(in) :: setup
                type(MeshDT), intent(in) :: mesh
                type(OutputDT), intent(inout) :: output
                type(ReturnsDT), intent(inout) :: returns
                type(Checkpoint_VariableDT), intent(in) :: checkpoint_variable
                integer, intent(in) :: time_step

                integer :: i, k, time_step_returns

                do i = 1, mesh%ng
                    k = mesh%rowcol_to_ind_ac(mesh%gauge_pos(i, 1), mesh%gauge_pos(i, 2))
                    output%response%q(i, time_step) = checkpoint_variable%ac_qz(k, setup%nqz)

                end do

                !$AD start-exclude
                if (allocated(returns%mask_time_step)) then
                    if (returns%mask_time_step(time_step)) then
                        time_step_returns = returns%time_step_to_returns_time_step(time_step)

                        !% Return states
                        if (returns%rr_states_flag) then
                            do i = 1, setup%nrrs

                                call ac_vector_to_matrix(mesh, checkpoint_variable%ac_rr_states(:, i), &
                                & returns%rr_states(time_step_returns)%values(:, :, i))

                            end do

                        end if

                        !% Return discharge grid
                        if (returns%q_domain_flag) then
                            call ac_vector_to_matrix(mesh, checkpoint_variable%ac_qz(:, setup%nqz), &
                            & returns%q_domain(:, :, time_step_returns))
                        end if

                    end if
                end if
                !$AD end-exclude

            end subroutine store_time_step

        Why has this section of code been removed from the differentiation? Firstly, Tapenade was returning a warning 
        (for some reason) and secondly, quite simply, this section allows you to store intermediate results which 
        can be useful when doing a forward run, but do not influence the calculation of gradients in the adjoint model.

    - Handle OpenMP directives
        The Tapenade parser does not detect all `OpenMP <https://www.openmp.org/>`__ directives.
        This is why we patch this file ourselves. For example, in the file
        ``fcore/operator/md_gr_operator.f90`` file, you can see how the OpenMP directives are used.

        .. code-block:: fortran
            
                    beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp
            #ifdef _OPENMP
                    !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
                    !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_ci, ac_cp, beta, ac_ct, ac_kexc, ac_hi, ac_hp, ac_ht, &
                    !$OMP& ac_qt) &
                    !$OMP& private(row, col, k, pn, en, pr, perc, l, prr, prd, qr, qd)
            #endif
                    do col = 1, mesh%ncol
                        do row = 1, mesh%nrow

        The directive pair ``#ifdef _OPENMP`` and ``#endif`` allows to enable or disable the OpenMP directives
        if the ``-fopenmp`` flag is passed to the compiler. However, Tapenade can not parse this. So if we want
        to generate an adjoint file with OpenMP directives, we patch as follows:

        .. code-block:: fortran
        
                beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp

                !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
                !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_ci, ac_cp, beta, ac_ct, ac_kexc, ac_hi, ac_hp, ac_ht, &
                !$OMP& ac_qt) &
                !$OMP& private(row, col, k, pn, en, pr, perc, l, prr, prd, qr, qd)

                do col = 1, mesh%ncol
                    do row = 1, mesh%nrow

        If we want to generate an adjoint file without OpenMP directive, we patch as follows: 

        .. code-block:: fortran
                
                beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp

                do col = 1, mesh%ncol
                    do row = 1, mesh%nrow

- ``Generate tapenade file``
    This step calls the Tapenade executable (supplied with the code sources) and generates the adjoint file
    from the source files.

    .. code-block:: python

        tapenade_exe = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "tapenade_3.16", "bin", "tapenade"
        )
        files = [os.path.join(".", os.path.basename(f)) for f in fortran_files]
        cmd_args = [
            tapenade_exe,
            "-b",
            "-d",
            "-fixinterface",
            "-noisize",
            "-context",
            "-msglevel",
            "100",
            "-adjvarname",
            "%_b",
            "-tgtvarname",
            "%_d",
            "-o",
            module,
            "-head",
            r"base_forward_run(parameters.control.x)\(output.cost)",
        ]

        if openmp:
            cmd_args.append("-openmp")

        cmd_args.extend(files)

        subprocess.run(cmd_args, check=True)

    It is possible to find in the Tapenade online documentation 
    `here <https://tapenade.gitlabpages.inria.fr/userdoc/build/html/tapenade/tutorial.html>`__ or by running the executable with the ``-h`` option,
    information to understand what is done through the various options.

    .. code-block:: shell

        ./smash/tapenade/tapenade_3.16/bin/tapenade -h
        Tapenade 3.16 (master) -  9 Oct 2020 17:47 - Java 11.0.21 Linux
        @@ TAPENADE_HOME=/home/fcolleoni/Documents/git/smash-repo/smash/tapenade/tapenade_3.16/bin/..
        Builds a differentiated program.
        Usage: tapenade [options]* filenames
        options:
        -head, -root <proc>     set the differentiation root procedure(s)
                                See FAQ for refined invocation syntax, e.g.
                                independent and dependent arguments, multiple heads...
        -tangent, -d            differentiate in forward/tangent mode (default)
        -reverse, -b            differentiate in reverse/adjoint mode
        ...
        -version                display Tapenade version information
        Report bugs to <tapenade@lists-sop.inria.fr>.

    There are still a few mysteries and sometimes it's necessary to check into the code examples, available on Tapenade's GitLab
    `here <https://gitlab.inria.fr/tapenade/tapenade>`__.
    
    In order to simplify this process, all the options are briefly detailed below.

    - ``-b``
        To differentiate in the reverse mode (adjoint model)

    - ``-d``
        To differentiate in the tangent mode (linear tangent model)

    - ``-fixinterface``
        To disable the use of activity to filter user-given (in)dependent vars

    - ``-noisize``
        To allow the use of dynamic calls to Fortran SIZE primitive whenever Tapenade needs the size of a variable

    - ``-openmp``
        To use the OpenMP directives to generate a parallel adjoint model
    
    - ``-context``
        To generate a complete differentiated code with its main procedure. This option is mandatory when the ``-openmp`` option is used
    
    - ``-msglevel 100``
        To set the level of detail of error messages (``100`` is the max)

    - ``-adjvarname %_b``
        To set the extension for adjoint variables
    
    - ``-tgtvarname %_d``
        To set the extension for linear tangent variables

    - ``-o module``
        To set the name of the generated file. The name of the file will be ``<module>_db.f90``
    
    - ``-head "base_forward_run(parameters.control.x)\(output.cost)"``
        To set the differentiation root procedure. ``base_forward_run`` is the **top differentiation routine** to differentiate,
        ``output.cost`` the **dependent output** variable whose derivate is required and
        ``parameters.control.x`` the **independent input** variable with respect to which differentiation must be made

- ``Patch tapenade file``
    After calling the Tapenade executable, this step allows you to make changes to the files to be differentiated. The only modification
    currently available is the ability to change the derived type used. By default, Tapenade generates a new version of an existing derived type
    by adding the suffix ``_DIFF`` and by removing all the variables that do not interact in differentiation. 
    This may be useful to avoid storing variables unnecessarily, but it implies the use of specific derived types, which can only be used 
    by the routines in ``forward_db.f90`` and which make the code more complicated to use for very little. Most of the variables that take up 
    memory are used in the differentiation scheme. For example, in the file ``fcore/forward/forward_db.f90`` file,
    the following derived type was patched as follows:

    .. code-block:: fortran

        TYPE(PARAMETERSDT), INTENT(INOUT) :: parameters
        TYPE(PARAMETERSDT_DIFF), INTENT(INOUT) :: parameters_b

    becomes

    .. code-block:: fortran

        TYPE(PARAMETERSDT), INTENT(INOUT) :: parameters
        TYPE(PARAMETERSDT), INTENT(INOUT) :: parameters_b

Tapenade tips
'''''''''''''

Here's a list of some useful tips when using Tapenade:

- Use simple Fortran 90 functionalities. Don't get lost in trying to make the code more complex in order to modularise it or remove duplicated code,
  this generally leads to the use of functionality that is not taken into account in Tapenade.

- At each generation of the adjoint model, a file containing potential error messages is available, ``smash/fcore/forward/forward_db.msg``. 
  As soon as an error occurs, consult the dedicated section in the Tapenade documentation,
  `here <https://tapenade.gitlabpages.inria.fr/userdoc/build/html/tapenade/faq.html#allMsgs>`__

Python guideline
----------------

.. warning::
    
    Section in development

Test
----

Tests are run with the ``make test`` or ``make test-coverage`` command using the `pytest <https://docs.pytest.org/en/7.4.x/>`__ library:

- ``make test``
    Run unit tests. This tests are also run in the continuous integration service (``CI``).

- ``make test-coverage``
    Run unit tests with coverage. It will display coverage result in the terminal and generate a html file.

    .. note::
        The html file can be viewed with your browser

        .. code-block:: none

            firefox smash/tests/htmlcov/index.html

There are two types of test available in `smash`:

- ``standard test``
    Test which do not require comparison with a file of expected values

- ``baseline test``
    Test that require a comparison with a file of expected values (e.g. ``smash/tests/baseline.hdf5``, 
    ``smash/tests/simulated_discharge.hdf5``)

Standard test
*************

To set up a standard test, all you need to do is add a function whose name starts with ``test_``, either from a pre-existing file or 
from a new file whose name must also start with ``test_``. Then all you have to do is write the desired tests and check the result with the 
``assert`` command

.. code-block:: python

    def test_add_two():
        x = 2
        y = -2

        assert add_two(x) == 4, "add_two.x"
        assert add_two(y) == 0, "add_two.y"


Baseline test
*************

Setting up a test with a comparison with an expected value is a little more complex than a standard test. It breaks down into
into two functions:

- ``generic function``
    This function is not a test function in itself, it simply runs the calculations and stores the variables to be checked.
    This function can take any kind of arguments as input but must returns a dictionary.

- ``test function``
    This function is the test function, which uses the ``generic function`` to generate the values to be compared and then compares them
    with a file in which the expected values have been stored.

.. code-block:: python

    def generic_add_sth_complex(**kwargs):

        x = 3
        y = [-2, 2]

        res = {
            "add_sth_complex.x": add_sth_complex(x),
            "add_sth_complex.y": add_sth_complex(y),
        }
        return res

    def test_add_sth_complex():

        res = generic_add_sth_complex()

        for key, value in res.items():
            assert value, pytest.baseline[key], key

In this example, we can't simply write what the result of ``add_sth_complex`` (because it's something complex). So we store the output value(s) of
this function in a file ``baseline.hdf5`` and then compare this value(s) by rerunning the test with the same function.

As you can see, we compare the values with values stored in ``pytest.baseline``. It is possible with pytest to store 
global variables at test runtime. This is done in the ``test_define_global_vars.py`` file and ``pytest.baseline`` is therefore the global variable
which stores data from the ``baseline.hdf5`` file and which can be called in any function.

Now comes the time when changes have been made to the code and the ``add_sth_complex`` function has been modified and still returns something 
complex but different. If we run the tests again, they will fail because the expected value is not up to date. It is therefore necessary to 
regenerate the expected values file (``baseline.hdf5``). To do this, you need to run the ``make test-baseline`` command, which will run the
``smash/tests/gen_baseline.py`` file, updating the ``baseline.hdf5`` file (by calling all the functions starting with ``generic_``), writes a
``diff_baseline.csv`` file which logs the differences between the old and new baselines and generates a new baseline file ``new_baseline.hdf5``.
If the logs in the ``diff_baseline.csv`` file seem consistent with your modifications (i.e. that tests that shouldn't be modified aren't modified
and conversely that tests that should be modified are modified), all you have to do is simply delete the ``baseline.hdf5`` file and rename
``new_baseline.hdf5`` to ``baseline.hdf5``.

.. note::

    To properly generate the ``diff_baseline.csv`` file, please ensure that you have the ``baseline.hdf5`` file from the latest commit
    on the main branch before running the ``make test-baseline`` command.

Documentation
-------------

The `smash` documentation is generated with the ``make doc`` command using `Sphinx <https://www.sphinx-doc.org/en/master/>`__. 
This command will generate a ``build/html`` folder in which it is possible to display the documentation on your browser.

.. code-block:: shell

    firefox ./doc/build/html/index.html

.. note::

    If you encounter any issues when compiling the documentation, try cleaning the ``doc`` directory and then recompiling the documentation.
    This can help eliminate any potential conflicts and bugs that may be causing the issue.

    .. code-block:: shell

        (smash-dev) make doc-clean
        (smash-dev) make doc

Generate a new ReStructuredText file
************************************

The documentation is written using files in `ReStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/>`__ (``rst``) format. 
It is possible to generate a new file by hand without too much difficulty, but the ``doc/source/gen_rst.py`` file makes it easier to create a new 
one depending on where it is placed in the documentation architecture.

.. code-block:: shell

    python3 gen_rst.py user_guide/quickstart/foo.rst

This will create the ``foo.rst`` file in the desired location with the following header:

.. code-block:: rst

    .. _user_guide.quickstart.foo:

    ===
    Foo
    ===

Then, you need to call up this file in the desired toctree, for example in the file ``doc/source/user_guide/index.rst``.

.. code-block:: rst

    .. _user_guide:

    ==========
    User Guide
    ==========

    Quickstart
    ----------
    .. toctree::
        :maxdepth: 1

        quickstart/cance_first_simulation

        quickstart/foo

User guide
**********

The user guide contains all the `smash` tutorials. These tutorials are not hardcoded, the python commands written in the
``.. ipython:: python`` directives are executed and automatically generate the tutorial output. This is quite handy, as it means you don't have to 
update the documentation each time the source is modified, and adds an extra layer of testing since the documentation will not compile if 
there is an error in executing a python command but which, on the other hand, requires a certain amount of computing time.

API reference
*************

Only the architecture of this section is defined in the ``rst`` files. The content is automatically generated from the docstrings of each
`smash` function. The style guide used for the docstrings is that of  `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`__.