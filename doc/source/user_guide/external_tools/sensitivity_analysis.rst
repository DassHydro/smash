.. _user_guide.external_tools.sensitivity_analysis:

.. _For documentation in external tools, it should be made with pre-existing output, and will not be reruned during compilation.

====================
Sensitivity Analysis
====================

In this tutorial, we will show how to perform sensitivity analysis on a
single catchment using ``SMASH`` and ``SALib``.

Sensitivity analysis on a single catchment
------------------------------------------

In this example, we will use Cance catchment to play around.

Import the necessary libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import smash
    import pandas as pd
    import numpy as np
    from datetime import date
    from tqdm import tqdm
    from SALib.analyze.sobol import analyze
    from SALib.sample.sobol import sample

Define the catchment, load the model and data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The setup and the mesh corresponding to the catchment.

.. code:: ipython3

    setup, mesh = smash.factory.load_dataset("cance")

Create the model object. This also loads the data into the model object.

.. code:: ipython3

    model = smash.Model(setup, mesh)


.. parsed-literal::

    </> Reading precipitation: 100%|██████████| 1440/1440 [00:05<00:00, 280.60it/s]
    </> Reading daily interannual pet: 100%|██████████| 366/366 [00:00<00:00, 811.93it/s] 
    </> Disaggregating daily interannual pet: 100%|██████████| 1440/1440 [00:00<00:00, 40575.31it/s]


.. parsed-literal::

    </> Computing mean atmospheric data
    </> Adjusting GR interception capacity


Generate the samples
~~~~~~~~~~~~~~~~~~~~

A problem is defined by the number of parameters, the names of the
parameters, and the bounds of the parameters. Please refer to the `SALib
documentation <https://salib.readthedocs.io/en/latest/api.html>`__ for
more examples on how to define a problem.

Sample size is the number of distinct values for each parameter.

.. code:: ipython3

    problem = {
        "num_vars": 4,
        "names": ["cp", "ct", "kexc", "llr"],
        "bounds": [(1, 3000), (1, 4000), (-25, 5), (1, 200)],
    }
    
    sample_size = 1024

Generate the sample using SAlib’s Satelli method. These are the samples
that will be used for the sensitivity analysis.

.. code:: ipython3

    param_values = sample(problem, sample_size, seed=1, calc_second_order=False)
    param_values




.. parsed-literal::

    array([[ 467.24048835, 2355.40055781,   -6.77403474,   49.23759794],
           [2537.22270641, 2355.40055781,   -6.77403474,   49.23759794],
           [ 467.24048835, 1050.97977774,   -6.77403474,   49.23759794],
           ...,
           [ 465.07810281, 1061.24262134,   -8.39497082,    6.94480178],
           [ 465.07810281, 1061.24262134,  -10.48251348,  141.57036807],
           [2627.85471268,  468.67423326,   -8.39497082,  141.57036807]],
          shape=(6144, 4))



Run the model on the chosen samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use smash.multiple_forward_run to get output from all samples quickly.

We define a function ``run_with_params`` to do the forward run from a
set of parameter to outputs that include all the metrics or signatures
that you want to analyze post-SMASH. In this case, I choose to include
NSE - a classical hydrological metric, Crc - continuous runoff
coefficients, and Eff - flood flow.

.. code:: ipython3

    def run_with_params(params):
        model.set_rr_parameters('cp', params[0])
        model.set_rr_parameters('ct', params[1])
        model.set_rr_parameters('kexc', params[2])
        model.set_rr_parameters('llr', params[3])
        model.forward_run()
        signatures = smash.signatures(model, sign=['Crc', 'Eff'], domain='sim')
        crc = signatures.cont.iloc[0]['Crc']
        eff = signatures.event.iloc[0]['Eff']
        nse = smash.evaluation(model, metric='nse')[0][0]
    
        return nse, crc, eff


Run the function for all the samples.

.. code:: ipython3

    %%capture  
    # Suppress the output of this cell, many forward runs text will be printed
    
    output = []
    for i in range(param_values.shape[0]):
        output.append(np.array(run_with_params(param_values[i])))

.. code:: ipython3

    output = np.array(output)
    
    Y_nse = np.array(output[:, 0])
    Y_crc = np.array(output[:, 1])
    Y_eff = np.array(output[:, 2])

Normalize the NSE. The normalized NSE is calculated as: $
:raw-latex:`\text{NNSE}` = :raw-latex:`\frac{1}{2 - \text{NSE}}` $

This normalized NSE projects the NSE metric from $
[-:raw-latex:`\infty`, 1] $ to $ [0, 1] $ in a way that preserve the
valuable information on good forward run while diminishing the impact of
bad forward run on the sensitivity analysis. That is why we use the
normalized NSE for this analysis.

.. code:: ipython3

    Y_nnse = 1/(2 - Y_nse)

Perform the sensitivity analysis using SAlib and show the results.

.. code:: ipython3

    Si_nnse = analyze(problem, Y_nnse, print_to_console=False, calc_second_order=False)
    print('--- First order sensitivity analysis on NSE ---')
    print('Sensitivity indices: ', Si_nnse['S1'])
    print('Confidence intervals: ', Si_nnse['S1_conf'])
    
    Si_crc = analyze(problem, Y_crc, print_to_console=False, calc_second_order=False)
    print('--- First order sensitivity analysis on CRC ---')
    print('Sensitivity indices: ', Si_crc['S1'])
    print('Confidence intervals: ', Si_crc['S1_conf'])
    
    Si_eff = analyze(problem, Y_eff, print_to_console=False, calc_second_order=False)
    print('--- First order sensitivity analysis on Eff ---')
    print('Sensitivity indices: ', Si_eff['S1'])
    print('Confidence intervals: ', Si_eff['S1_conf'])


.. parsed-literal::

    --- First order sensitivity analysis on NSE ---
    Sensitivity indices:  [ 0.55052169  0.20563381 -0.02693246  0.02273316]
    Confidence intervals:  [0.24807971 0.28165997 0.0704906  0.01984534]
    --- First order sensitivity analysis on CRC ---
    Sensitivity indices:  [1.55958585e-02 3.28269253e-01 4.15976060e-03 2.44881492e-06]
    Confidence intervals:  [5.54563750e-01 3.00143536e-01 3.50225476e-01 5.25396648e-04]
    --- First order sensitivity analysis on Eff ---
    Sensitivity indices:  [0.40991146 0.0829772  0.01993396 0.00846561]
    Confidence intervals:  [0.25203485 0.15304919 0.04074188 0.01039334]


.. parsed-literal::

    /local/AIX/nbnguyen/miniconda3/envs/smash-dev/lib/python3.13/site-packages/SALib/util/__init__.py:274: FutureWarning: unique with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated and will raise in a future version.
      names = list(pd.unique(groups))

