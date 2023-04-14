.. _user_guide.in_depth.multiple_run:

============
Multiple run
============

Here, we aim to compute multiple forward runs of a single Model object (in parallel) in order to retrieve the cost (and discharge values) for each parameters set tested.
This method is often used to perform global sensitivity analysis.

To get started, open a Python interface:

.. code-block:: none

    python3
    
-------
Imports
-------

.. ipython:: python
    
    import smash
    import numpy as np
    import time

---------------------
Model object creation
---------------------

First, you need to create a :class:`smash.Model` object. 
For this case, we will use the ``Lez`` dataset.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.load_dataset("Lez")
    
    model = smash.Model(setup, mesh)

------------------------
Generation of the sample
------------------------

In order to compute several forward runs of the :class:`smash.Model` object, we first need to generate a ``sample`` of parameters.
We define a custom ``sample`` with random sets of parameters using the :meth:`smash.generate_samples` method.

First, we refer to the :meth:`smash.Model.get_bound_constraints` method to obtain some information about the Model parameters:

.. ipython:: python

    model.get_bound_constraints()

Then, we will define a ``problem`` based on the previous information by adjusting the bounds of the parameters:

.. ipython:: python

    problem = {
        "num_vars": 4,
        "names": ["cp", "cft", "exc", "lr"],
        "bounds": [[1, 2000], [1, 1000], [-20, 5], [1, 1000]]
    }

Once the ``problem`` is defined we can generate the ``sample``. Here we define a ``sample`` of 400 random sets of Model parameters:

.. ipython:: python

    sample = smash.generate_samples(problem, n=400, random_state=1)

.. note::
    **random_state** argument is used to set the random seed.

***************************
Visualization of the sample
***************************

``sample`` is a :class:`SampleResult <smash.SampleResult>` object with several methods including the :meth:`SampleResult.slice() <smash.SampleResult.slice>` method.
This method allows to create slices of the original sample (sub-sample). We will use this method to quickly visualize the ``sample``.

Visualization of the 10 first sets:

.. ipython:: python

    sample.slice(10)

We can also visualize sets between ``start`` and ``end`` index:

.. ipython:: python

    sample.slice(start=10, end=20)

************************
Conversion of the sample
************************

Two other methods can be used to convert this object to pandas.DataFrame or numpy.ndarray. The :meth:`SampleResult.to_dataframe() <smash.SampleResult.to_dataframe()>` method and
:meth:`SampleResult.to_numpy() <smash.SampleResult.to_numpy()>`

.. ipython:: python

    sample.to_dataframe()

    # axis=-1 to stack along columns
    sample.to_numpy(axis=-1)

--------------------------------
Computation of the multiple runs
--------------------------------

Once the ``sample`` is generated, the mutliple runs can be simulated using the :meth:`Model.multiple_run <smash.Model.multiple_run>` method.

********************************
Multiple runs with default setup
********************************

Here we will compute the multiple runs with the default setup. That is, returning the value of the cost function for each set using 1 CPU.
The default cost function is a ``nse`` calculated on the most downstream gauge.

.. ipython:: python

    mtprr = model.multiple_run(sample)

``mtprr`` is a :class:`MultipleRunResult <smash.MultipleRunResult>` object which has a ``cost`` attribute. We can visualize the 10 first cost values returned.

.. ipython:: python

    mtprr.cost[0:10]

***********************************************
Multiple runs with simulated discharge returned
***********************************************

We can return the simulated discharge for each set by setting ``True`` to the **return_qsim** argument.

.. ipython:: python

    mtprr_qsim = model.multiple_run(sample, return_qsim=True)

This time, ``mtprr_qsim`` is a :class:`MultipleRunResult <smash.MultipleRunResult>` object which has a ``cost`` and ``qsim`` attribute. We can visualize the 10 first cost and simulated
discharge (on the most downstream gauge) values returned.

.. ipython:: python

    mtprr_qsim.cost[0:10]
    # The shape correspond to (number of gauges, number of time steps, number of sets)
    mtprr_qsim.qsim.shape
    mtprr_qsim.qsim[0,:,0:10]

*****************************************
Multiple runs with customize cost options
*****************************************

We can also change the cost option arguments such as the kind of objective function (**jobs_fun** argument) and on which gauge we want to compute the cost (**gauge** argument).

.. ipython:: python

    mtprr_cost_opt = model.multiple_run(sample, jobs_fun="kge", gauge="all")

We can visualize the 10 first cost values returned.

.. ipython:: python

    mtprr_cost_opt.cost[0:10]

*************************
Multiple runs in parallel
*************************

At the moment, all the previous runs were done sequentially. we can save computation time by parallelizing the runs on several CPUs by assigning a value to the **ncpu** argument.
We will used the ``time`` library, previously imported, to retrieve the computation time and compare it between a sequential and parallel run.

.. ipython:: python

    start_seq = time.time()

    mtprr_seq = model.multiple_run(sample)

    time_seq = time.time() - start_seq

    start_prl = time.time()

    mtprr_prl = model.multiple_run(sample, ncpu=4)

    time_prl = time.time() - start_prl

We can now visualize that we have the same data between a sequential and parallel run and the difference of time computation in seconds.

.. ipython:: python

    mtprr_seq.cost[0:10]
    mtprr_prl.cost[0:10]

    time_seq, time_prl

***************************
Multiple runs with iterator
***************************

In case you want to retrieve the simulated discharge for each set of the sample, it may be impossible to store the entire data in memory.
As a reminder, the simulated discharge data is stored as single precision floating point (``float32``) in an array of size :math:`ng \times nt \times ns` where :math:`ng` is the number of gauges,
:math:`nt` the number of time steps and :math:`ns` the number of sets in the sample.
In our case, it results in an array of size :math:`n=3 \times 364 \times 400 = 436 800`. One way to get around this problem is to iterate on sub-samples and to serialize each output to
free the memory.

To perform this we will use the :meth:`SampleResult.iterslice() <smash.SampleResult.iterslice()>` method to iterate on the sample.
This method as a **by** argument, set to 1 by default, which allows the user to choose the size of the slice (sub-sample) to iterate on.
We will fix the **by** argument to 100 to perform 4 iterations on a sample of size 400.
In this way, we will divide by 4 the memory space taken by the array of simulated discharge.

.. ipython:: python
    :verbatim:

    for i, slc in enumerate(sample.iterslice(by=100)):
        mtprr = model.multiple_run(slc, ncpu=4, return_qsim=True)
        smash.save_model_ddt(
            model,
            f"res_mtprr_slc_{i}.hdf5",
            sub_data={"mtprr_cost": mtprr.cost, "mtprr_qsim": mtprr.qsim}
        )

.. note::
    We used the :meth:`smash.save_model_ddt` method to serialize the result on each iteration by filling in the **sub_data** argument.
