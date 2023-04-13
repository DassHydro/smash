.. _user_guide.in_depth.optimize.multi_criteria:

===========================
Multi-criteria optimization
===========================

Here, we aim to optimize the Model parameters/states using multiple calibration metrics based on certain hydrological signatures.

To get started, open a Python interface:

.. code-block:: none

    python3
    
-------
Imports
-------

.. ipython:: python
    
    import smash
    import matplotlib.pyplot as plt

---------------------
Model object creation
---------------------

First, you need to create a :class:`smash.Model` object. 
For this case, we will use the ``Lez`` dataset.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.load_dataset("Lez")
    
    model = smash.Model(setup, mesh)

---------------------------------------------
Multiple metrics calibration using signatures
---------------------------------------------

This method enables the incorporation of multiple calibration metrics into the observation term of the cost function (for mathematical details, see the :ref:`Math / Num Documentation <math_num_documentation.signal_analysis.cost_functions>` section).
:ref:`Hydrological signatures <math_num_documentation.signal_analysis.hydrological_signatures>` are thus introduced to tackle such an approach. 
Note that this multi-criteria approach is possible for all optimization methods, including :meth:`smash.Model.optimize`, :meth:`smash.Model.bayes_estimate`, :meth:`smash.Model.bayes_optimize` and :meth:`smash.Model.ann_optimize`. 
For simplicity, in this example, we use :meth:`smash.Model.optimize` with a uniform mapping.

Let us consider a classical calibration with a single metric:

.. ipython:: python

    model_sm = model.optimize(jobs_fun="nse");

Now we employ, for instance, continuous and flood-event runoff coefficients (``Crc`` and ``Erc``) for multi-criteria calibration:

.. ipython:: python

    model_mm = model.optimize(jobs_fun=["nse", "Crc", "Erc"], wjobs_fun=[0.6, 0.1, 0.3]);

where the weights of the objective functions based on ``nse``, ``Crc``, ``Erc`` are set to 0.6, 0.1 and 0.3 respectively. 
If these weights are not given by user, the cost value is computed as the mean of the objective functions.

For multiple metrics calibration based on flood-event signatures, we can further adjust some parameters in the :ref:`segmentation <user_guide.in_depth.event_segmentation>` algorithm to compute flood-event signatures. 
For example, we use a multi-criteria cost function based on the peak flow ``Epf`` to calibrate the Model parameters:

.. ipython:: python
    :suppress:

    model_mme = model.optimize(
            jobs_fun=["nse", "Epf"], 
            event_seg={"peak_quant": 0.99}, 
            wjobs_fun=[0.6, 0.4]
        )

.. ipython:: python
    :verbatim:

    model_mme = model.optimize(
            jobs_fun=["nse", "Epf"], 
            event_seg={"peak_quant": 0.99}, 
            wjobs_fun=[0.6, 0.4]
        )

Finally, the simulated discharges of the three models can be visualized as follows:

.. ipython:: python

    qo = model.input_data.qobs[0,:].copy()
    qo = np.where(qo<0, np.nan, qo)  # to deal with missing data
    plt.plot(qo, label="Observed discharge");
    plt.plot(model_sm.output.qsim[0,:], label="Simulated discharge - sm");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model_mm.mesh.code[0]);
    @savefig user_guide.in_depth.optimize.multi_criteria.qsim_sm.png
    plt.legend();

.. ipython:: python

    plt.plot(qo, label="Observed discharge");
    plt.plot(model_mm.output.qsim[0,:], label="Simulated discharge - mm");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model_mm.mesh.code[0]);
    @savefig user_guide.in_depth.optimize.multi_criteria.qsim_mm.png
    plt.legend();

.. ipython:: python

    plt.plot(qo, label="Observed discharge");
    plt.plot(model_mme.output.qsim[0,:], label="Simulated discharge - mme");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model_mm.mesh.code[0]);
    @savefig user_guide.in_depth.optimize.multi_criteria.qsim_mme.png
    plt.legend();

.. ipython:: python
    :suppress:

    plt.close('all')