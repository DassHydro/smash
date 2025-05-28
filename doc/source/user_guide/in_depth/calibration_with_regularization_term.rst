.. _user_guide.in_depth.calibration_with_regularization_term:

====================================
Calibration with Regularization Term
====================================

Preliminaries
-------------

We start by importing the modules needed in this tutorial.

.. code-block:: python
	
	>>> import smash
	>>> import matplotlib.pyplot as plt
	>>> import numpy as np

Now, we need to create a :class:`smash.Model` object.
For this case, we use the :ref:`user_guide.data_and_format_description.cance` dataset as an example.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. code-block:: python
	
	>>> setup, mesh = smash.factory.load_dataset("Cance")
	>>> model = smash.Model(setup, mesh)

Distributed calibration with regularisation
-------------------------------------------

The regularization function can be applied with a distributed mapping using the lbfgsb optimizer. That kind of calibration produces smoother parameters map with better correlation with the background.

To trigger the regularization function we must change the optimize and cost options:

.. code-block:: python

    >>> optimize_options = {"termination_crit": {"maxiter": 30}}

    >>> cost_options= {
    ...        "end_warmup": '2014-10-15 00:00',
    ...        "jobs_cmpt": "nse",
    ...        "wjobs_cmpt": "mean",
    ...        "jreg_cmpt": ["prior", "smoothing"],
    ...        "wjreg": 0.0,
    ...        }

The maximum of iteration has been increased so that enough iterations will be performed to reduce the cost (the criteria and the regularisation term). "end_warmup" has been set to one mont latter than the model start time to avoid to take into account "jumps" of the model outputs during the cost computation.

When triggering the regularisation, the total cost J is defined as follow:

.. math::
    \boxed{
    J = J0 + wjreg \times Jreg}

where :math:`J0` is the missfit criteria and :math:`Jreg` the regularisation term defined as follow:

.. math::
    \boxed{
    Jreg = wjreg\_cmpt_1 \times Jprior + wjreg\_cmpt_2 \times Jsmoothing}

where :math:`wjreg\_cmpt_1, wjreg\_cmpt_2, ...` are some weight coefficients attributed to each penalisation function.

"jreg_cmpt" option selects the penalisation function (choice are a list of ["prior", "smoothing", "hard_smothing"]) used to compute the regularisation term: "prior" denoted :math:`jprior` compute :math:`|| X - X^b ||` (norm of the distance of the control parameter :math:`X` to the background :math:`X^b`), "smoothing" and "hard_smoothing" are some spatial smoothing function denoted :math:`jsmoothing` (see :ref:`math_num_documentation.regularization_function` for more details). "wjreg" provide the total weight of the regularisation function (here it is set to 0., i.e the regularisation will be inefficient). 

Now we start the optimization of the model parameters.  

.. code-block:: python

    >>> model_noreg = smash.optimize(model, 
    ...                    optimizer="lbfgsb",
    ...                    mapping="distributed",
    ...                    optimize_options=optimize_options,
    ...                    cost_options=cost_options,
    ...                    )

That return:

.. code-block:: output
	
    </> Optimize
        At iterate     0    nfg =     1    J = 5.82505e-01    |proj g| = 2.26224e-02
        At iterate     1    nfg =     2    J = 4.98505e-01    |proj g| = 2.28896e-02
        At iterate     2    nfg =     4    J = 2.64331e-01    |proj g| = 3.06342e-02
        ...
        At iterate    28    nfg =    32    J = 1.48585e-02    |proj g| = 8.08515e-03
        At iterate    29    nfg =    33    J = 1.46492e-02    |proj g| = 5.14101e-03
        At iterate    30    nfg =    35    J = 1.45157e-02    |proj g| = 8.50714e-03
        STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Let's now plot the discharges and the map of the calibrated parameters.

.. code-block:: python

    >>> plt.plot(model_noreg.response_data.q[0, :], label="Observed discharge")
    >>> plt.plot(model_noreg.response.q[0, :], label="Simulated discharge")
    >>> plt.xlabel("Time step")
    >>> plt.ylabel("Discharge ($m^3/s$)")
    >>> plt.grid(ls="--", alpha=.7)
    >>> plt.legend()
    >>> plt.title(f"Observed and simulated discharge at gauge {code}")
    >>> plt.show()

.. image:: ../../_static/user_guide.in_depth.calibration_with_regularisation_term.hydrograph_noreg.png
    :align: center

.. code-block:: python

    >>> plt.imshow(model_noreg.rr_parameters.values[:,:,1]) ;
    >>> plt.colorbar(label="Production capacity");
    >>> plt.title("Cance - map of the calibrated production parameter (cp)");

.. image:: ../../_static/user_guide.in_depth.calibration_with_regularisation_term.calibrated_noreg.png
    :align: center

Now change the weight of the regularisation term and see the effect.

.. code-block:: python

    >>> cost_options= {
    ...        "end_warmup": '2014-10-15 00:00',
    ...        "jobs_cmpt": "nse",
    ...        "wjobs_cmpt": "mean",
    ...        "jreg_cmpt": ["prior", "smoothing"],
    ...        "wjreg": 0.000001,
    ...        }

    >>> model_reg = smash.optimize(model, 
    ...                    optimizer="lbfgsb",
    ...                    mapping="distributed",
    ...                    optimize_options=optimize_options,
    ...                    cost_options=cost_options,
    ...                    )

That return:

.. code-block:: output
	
    </> Optimize
    At iterate     0    nfg =     1    J = 5.82505e-01    |proj g| = 2.26224e-02
    At iterate     1    nfg =     2    J = 5.26515e-01    |proj g| = 3.61495e-02
    At iterate     2    nfg =     4    J = 5.19688e-01    |proj g| = 1.09659e-01
    ...
    At iterate    28    nfg =    32    J = 3.54707e-01    |proj g| = 1.05339e-02
    At iterate    29    nfg =    33    J = 3.54489e-01    |proj g| = 9.41573e-03
    At iterate    30    nfg =    34    J = 3.54246e-01    |proj g| = 8.71998e-03
    STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

The simulated and observed discharges and the calibrated parameter map (cp) are plotted bellow:

.. code-block:: python

    >>> plt.plot(model_reg.response_data.q[0, :], label="Observed discharge")
    >>> plt.plot(model_reg.response.q[0, :], label="Simulated discharge")
    >>> plt.xlabel("Time step")
    >>> plt.ylabel("Discharge ($m^3/s$)")
    >>> plt.grid(ls="--", alpha=.7)
    >>> plt.legend()
    >>> plt.title(f"Observed and simulated discharge at gauge {code}")
    >>> plt.show()

.. image:: ../../_static/user_guide.in_depth.calibration_with_regularisation_term.hydrograph_reg.png
    :align: center

.. code-block:: python

    >>> plt.imshow(model_reg.rr_parameters.values[:,:,1]) ;
    >>> plt.colorbar(label="Production capacity");
    >>> plt.title("Cance - map of the calibrated production parameter (cp)");

.. image:: ../../_static/user_guide.in_depth.calibration_with_regularisation_term.calibrated_cp_reg.png
    :align: center

The simulated discharge does not fit well with the observed discharge, but the parameter map is smoother. Indeed the regularisation adds some constraints for the optimisation and forces the parameters to be correlated with its neighbourgs and close to the background. In our case, the weight for the regularisation term is too big compare to the optimisation criteria (nse) and the converrgence is less efficient. The optimal regularisation weight must be estimated by the l-curve method (P. C. Hansen and D. P. O’Leary, “The use of the L-curve in the regularization of discrete ill-posed problems,” SIAM Journal on Scientific Computing, vol. 14, no. 6, pp. 1487–1503, 1993.).

An automatic l-curve computation is implemented in Smash and the optimal coefficent "wjreg" can be automatically estimated (M. Jay-Allemand, P. Javelle, and P.-A. Garambois, “Assimilation dans SMASH.” 2023.). To do so, just set the "wjreg" cost option to "lcurve". In order to plot the result of th l-curve, setup some optional returned options to get more outputs. Notice that the lcurve will performe six optimisation cycle (the number of cycle is currently fixed) and thus the computation can take a long time.


.. code-block:: python

    >>> cost_options= {
    ...        "end_warmup": '2014-10-15 00:00',
    ...        "jobs_cmpt": "nse",
    ...        "wjobs_cmpt": "mean",
    ...        "jreg_cmpt": ["prior", "smoothing"],
    ...        "wjreg": "lcurve",
    ...        }


    return_options= {
    ...            "time_step": "all",
    ...            "lcurve_wjreg": True,
    ...            "jreg": True,
    ...            "jobs": True,
    ...            "cost": True,
    ...            "control_vector": True,
    ...            "n_iter": True,
    ...        }

    >>> model_reg_lcurve = smash.optimize(model, 
    ...                    optimizer="lbfgsb",
    ...                    mapping="distributed",
    ...                    optimize_options=optimize_options,
    ...                    cost_options=cost_options,
    ...                    return_options=return_options,
    ...                    )

That return:

.. code-block:: output
	
    </> Optimize
    L-CURVE WJREG CYCLE 1
    At iterate     0    nfg =     1    J = 5.82505e-01    |proj g| = 2.26224e-02
    At iterate     1    nfg =     2    J = 4.98505e-01    |proj g| = 2.28896e-02
    At iterate     2    nfg =     4    J = 2.64331e-01    |proj g| = 3.06342e-02
    ...
    At iterate    28    nfg =    32    J = 1.48585e-02    |proj g| = 8.08515e-03
    At iterate    29    nfg =    33    J = 1.46492e-02    |proj g| = 5.14101e-03
    At iterate    30    nfg =    35    J = 1.45157e-02    |proj g| = 8.50714e-03
    STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
    L-CURVE WJREG CYCLE 2
    At iterate     0    nfg =     1    J = 5.82505e-01    |proj g| = 2.26224e-02
    At iterate     1    nfg =     2    J = 4.98778e-01    |proj g| = 2.28872e-02
    At iterate     2    nfg =     5    J = 3.18313e-01    |proj g| = 1.74505e-02
    ...
    At iterate    27    nfg =    35    J = 1.47322e-01    |proj g| = 1.16875e-02
    At iterate    28    nfg =    36    J = 1.46963e-01    |proj g| = 9.66862e-03
    At iterate    29    nfg =    37    J = 1.46561e-01    |proj g| = 1.85026e-02
    At iterate    30    nfg =    38    J = 1.46371e-01    |proj g| = 3.53880e-03
    STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

The l-curve can be plotted (jreg compare to the final cost):

.. image:: ../../_static/user_guide.in_depth.calibration_with_regularisation_term.lcurve.png
    :align: center

The red cross shows the the optimal weight of the regularisation term estimated by the lcurve (2.09e-8). The green cross shows an approximation of the optimal weight (4.46e-8) achieved by setting "wjreg" to the value "fast". In that latter case only tow optimisation cycle will be performed.

We notice that the optimisation is much better (lower final cost) and ours parameters are now spatially correlated and the spatial mean must be close to the background value.


.. code-block:: python

    >>> plt.plot(model_reg_lcurve.response_data.q[0, :], label="Observed discharge")
    >>> plt.plot(model_reg_lcurve.response.q[0, :], label="Simulated discharge")
    >>> plt.xlabel("Time step")
    >>> plt.ylabel("Discharge ($m^3/s$)")
    >>> plt.grid(ls="--", alpha=.7)
    >>> plt.legend()
    >>> plt.title(f"Observed and simulated discharge at gauge {code}")
    >>> plt.show()

.. image:: ../../_static/user_guide.in_depth.calibration_with_regularisation_term.hydrograph_noreg.png
    :align: center

.. code-block:: python

    >>> plt.imshow(model_reg_lcurve.rr_parameters.values[:,:,1]) ;
    >>> plt.colorbar(label="Production capacity");
    >>> plt.title("Cance - map of the calibrated production parameter");

.. image:: ../../_static/user_guide.in_depth.calibration_with_regularisation_term.calibrated_cp_reg_lcurve.png
    :align: center

A penalisation term with harder smoothing can be used. Just set "jreg_cmpt" to ["prior", "hard-smoothing"]. More over weightning betwween each penalisation term can be parametrised. Let's define define a weighning twice larger for the smoothing than for the prior:

.. code-block:: python

    >>> cost_options= {
    ...        "end_warmup": '2014-10-15 00:00',
    ...        "jobs_cmpt": "nse",
    ...        "wjobs_cmpt": "mean",
    ...        "jreg_cmpt": ["prior", "hard-smoothing"],
    ...        "wjreg_cmpt": [1., 2.],
    ...        "wjreg": "2.09e-8",
    ...        }

    >>> model_reg_lcurve_hard_smoothing_with_pond = smash.optimize(model, 
    ...                    optimizer="lbfgsb",
    ...                    mapping="distributed",
    ...                    optimize_options=optimize_options,
    ...                    cost_options=cost_options,
    ...                    return_options=return_options,
    ...                    )

That return:

.. code-block:: output
	
    At iterate     0    nfg =     1    J = 6.95010e-01    |proj g| = 1.66423e-02
    At iterate     1    nfg =     2    J = 6.51908e-01    |proj g| = 1.78277e-02
    At iterate     2    nfg =     4    J = 4.08855e-01    |proj g| = 5.22046e-02
    ...
    At iterate    28    nfg =    34    J = 1.94216e-01    |proj g| = 2.09031e-02
    At iterate    29    nfg =    35    J = 1.93498e-01    |proj g| = 1.87484e-02
    At iterate    30    nfg =    36    J = 1.92991e-01    |proj g| = 2.03715e-02
    STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

The parameter map will be even more smooth:

.. code-block:: python

    >>> plt.imshow(model_reg_lcurve.rr_parameters.values[:,:,1]) ;
    >>> plt.colorbar(label="Production capacity");
    >>> plt.title("Cance - map of the calibrated production parameter");

.. image:: ../../_static/user_guide.in_depth.calibration_with_regularisation_term.calibrated_cp_lcurve_hard_smoothing.png
    :align: center

