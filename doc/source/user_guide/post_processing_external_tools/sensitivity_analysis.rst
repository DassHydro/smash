.. _user_guide.post_processing_external_tools.sensitivity_analysis:

.. For documentation in external tools, it should be made with pre-existing output, and will not be rerun during compilation.

====================
Sensitivity Analysis
====================

In this tutorial, we will show how to perform sensitivity analysis on a
single catchment using `smash` and `SALib <https://salib.readthedocs.io/>`__. We will use the :ref:`user_guide.data_and_format_description.cance` dataset as an example in this tutorial.

First, open a Python interface:

.. code-block:: none

    python3

Imports
-------

We will first import the necessary libraries for this tutorial.

.. code-block:: python

    >>> import smash
    >>> import numpy as np
    >>> from SALib.analyze.sobol import analyze
    >>> from SALib.sample.sobol import sample

.. note::

    The library `SALib <https://salib.readthedocs.io/>`__ is a requirement for this tutorial.
    It is not installed by default but can be installed with pip as follows:

    .. code-block:: none

        pip install SALib

Define the catchment, load the model and data
---------------------------------------------

The setup and the mesh corresponding to the catchment.

.. code-block:: python

    >>> setup, mesh = smash.factory.load_dataset("cance")

Create the model object. This also loads the data into the model object.

.. code-block:: python

    >>> model = smash.Model(setup, mesh)

.. code-block:: output

    </> Computing mean atmospheric data
    </> Adjusting GR interception capacity

Generate the samples
--------------------

A problem of sensitivity analysis is defined by the number of parameters, the names of the
parameters, and the bounds of the parameters. Please refer to the `SALib
documentation <https://salib.readthedocs.io/en/latest/api.html>`__ for
more examples on how to define a problem.

Sample size is the number of distinct values for each parameter.

.. code-block:: python

    >>> problem = {
    ...     "num_vars": 4,
    ...     "names": ["cp", "ct", "kexc", "llr"],
    ...     "bounds": [(1, 3000), (1, 4000), (-25, 5), (1, 200)],
    ... }
    >>> 
    >>> sample_size = 1024

.. note::

    A larger ``sample_size`` improves the accuracy of sensitivity index estimates but also increases computational time.

Generate the samples, which will be used for sensitivity analysis, using the Saltelli sampling method implemented in SALib.

.. code-block:: python

    >>> param_values = sample(problem, sample_size, seed=1, calc_second_order=False)
    >>> param_values

.. code-block:: output

    array([[ 467.24048835, 2355.40055781,   -6.77403474,   49.23759794],
           [2537.22270641, 2355.40055781,   -6.77403474,   49.23759794],
           [ 467.24048835, 1050.97977774,   -6.77403474,   49.23759794],
           ...,
           [ 465.07810281, 1061.24262134,   -8.39497082,    6.94480178],
           [ 465.07810281, 1061.24262134,  -10.48251348,  141.57036807],
           [2627.85471268,  468.67423326,   -8.39497082,  141.57036807]])

.. code-block:: python

    >>> param_values.shape

.. code-block:: output

    (6144, 4)

In this example, we have 6144 sets of 4 parameters. The number of sets varies based on
the sample size, the number of parameters, and whether we want to include second order sensitivity.
Details can be found in the `SALib documentation <https://salib.readthedocs.io/en/latest/api.html>`__.

Run the model on the chosen samples
-----------------------------------

We define a function ``run_with_params``, that performs a forward run using a set of parameters to compute performance metrics and/or hydrological signatures based on simulated discharge. In this case, we use
NSE - a classical hydrological metric, Crc - continuous runoff
coefficients, and Eff - flood flow as examples.

For more information on the available signatures and indices, please refer
to the :ref:`api_reference.principal_methods.signal_analysis` section.

.. code-block:: python

    >>> def run_with_params(model, params):
    ...     model.set_rr_parameters('cp', params[0])
    ...     model.set_rr_parameters('ct', params[1])
    ...     model.set_rr_parameters('kexc', params[2])
    ...     model.set_rr_parameters('llr', params[3])
    ...     model.forward_run()
    ...     signatures = smash.signatures(model, sign=['Crc', 'Eff'], domain='sim')
    ...     crc = signatures.cont.iloc[0]['Crc']
    ...     eff = signatures.event.iloc[0]['Eff']
    ...     nse = smash.evaluation(model, metric='nse')[0][0]
    ... 
    ...     return nse, crc, eff

.. hint::

    Using ``common_options={'n_cpu': n}`` (with n based on your system configuration)
    in the `smash.Model.forward_run` function will help accelerate the computation.

Run the function for all the samples using a simple ``for`` loop.

.. code-block:: python

    >>> output = []
    >>> for i in range(param_values.shape[0]):
    ...     output.append(np.array(run_with_params(model, param_values[i])))

.. hint::

    Each iteration calls the ``run_with_params`` function, which calls the `smash.Model.forward_run` function.
    Each ``forward_run`` prints a line of text, which is a lot of redundant text considering the number of iterations.
    You can suppress these outputs by redirecting them to a ``StringIO`` object. For example:

    .. code-block:: python

        >>> from contextlib import redirect_stdout
        >>> import io
        >>> 
        >>> def run_with_params(model, params):
        ...     # Redirect stdout to a null stream
        ...     with redirect_stdout(io.StringIO()):
        ...         # Set the parameters
        ...         print("This won't be displayed")
        ...         model.forward_run() # The output text in this function also won't be displayed
        ...         # The rest of the function

    However, this trick is beyond the scope of this tutorial, so it is just a tip, not a requirement.

Take out the 3 outputs array from the list.

.. code-block:: python

    >>> output = np.array(output)
    >>> 
    >>> Y_nse = np.array(output[:, 0])
    >>> Y_crc = np.array(output[:, 1])
    >>> Y_eff = np.array(output[:, 2])

Normalize the NSE. The normalized NSE is calculated as:

.. math::

   \text{NNSE} = \frac{1}{2 - \text{NSE}}

This normalized NSE maps the NSE metric from :math:`[-\infty, 1]` to :math:`[0, 1]`
in a manner that preserves valuable information on effective forward runs
while reducing the influence of ineffective runs on the sensitivity analysis.
This is why we utilize the normalized NSE for this analysis.

.. code-block:: python

    >>> Y_nnse = 1/(2 - Y_nse)

Perform the sensitivity analysis
--------------------------------

Now that the problem and their outputs are defined, we can perform
the sensitivity analysis using SALib and show the results.

.. code-block:: python

    >>> Si_nnse = analyze(problem, Y_nnse, print_to_console=False, calc_second_order=False, seed=1)
    >>> print('</> First order sensitivity analysis on NSE')
    >>> print('    Sensitivity indices: ', Si_nnse['S1'])
    >>> print('    Confidence intervals: ', Si_nnse['S1_conf'])
    >>> 
    >>> Si_crc = analyze(problem, Y_crc, print_to_console=False, calc_second_order=False, seed=1)
    >>> print('</> First order sensitivity analysis on Crc')
    >>> print('    Sensitivity indices: ', Si_crc['S1'])
    >>> print('    Confidence intervals: ', Si_crc['S1_conf'])
    >>> 
    >>> Si_eff = analyze(problem, Y_eff, print_to_console=False, calc_second_order=False, seed=1)
    >>> print('</> First order sensitivity analysis on Eff')
    >>> print('    Sensitivity indices: ', Si_eff['S1'])
    >>> print('    Confidence intervals: ', Si_eff['S1_conf'])

.. code-block:: output

    </> First order sensitivity analysis on NSE
        Sensitivity indices:  [ 0.55052169  0.20563381 -0.02693246  0.02273316]
        Confidence intervals:  [0.23071231 0.27603189 0.08007255 0.02181141]
    </> First order sensitivity analysis on Crc
        Sensitivity indices:  [1.55958585e-02 3.28269253e-01 4.15976060e-03 2.44881492e-06]
        Confidence intervals:  [0.55138231 0.29921365 0.26705503 0.00057764]
    </> First order sensitivity analysis on Eff
        Sensitivity indices:  [0.40991146 0.0829772  0.01993396 0.00846561]
        Confidence intervals:  [0.24355879 0.13125132 0.04128444 0.01077616]

The sensitivity indices show the relative importance of each parameter in
affecting the model outputs. A higher sensitivity index indicates that the
parameter has a stronger influence on that particular metric. The confidence
intervals provide a measure of uncertainty in these sensitivity estimates.

.. note::

    This analysis is performed on a single catchment. You can also perform this
    analysis on multiple catchments by doing the same in a loop. Specifying the
    ``ncpu`` or using multiprocessing could help reduce the run time.
