.. _user_guide.optimize.pre_regio_ann:

===================================================
Pre-regionalization using artificial neural network
===================================================

Here, we aim to employ some physiographic descriptors to find the pre-regionalization mapping using an artificial neural network. 
Six descriptors are considered in this example, which are:

- Slope
- Drainage density
- Karst
- Woodland
- Urban
- Soil water storage

First, open a Python interface:

.. code-block:: none

    python3
    
-------
Imports
-------

.. ipython:: python
    
    import smash
    import matplotlib.pyplot as plt
    import numpy as np

---------------------
Model object creation
---------------------

To perform the calibrations, you need to create a :class:`smash.Model` object. 
For this case, we will use the ``Lez`` dataset.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.load_dataset("Lez")
    
    model = smash.Model(setup, mesh)

----------------------------
Visualization of descriptors
----------------------------

This method requires input descriptors, which were provided during the creation of the Model object. 
We can visualize these descriptors and verify if they were successfully loaded:

.. ipython:: python

    model.input_data.descriptor.shape

.. ipython:: python

    desc_name = model.setup.descriptor_name
    fig, axes = plt.subplots(1, len(desc_name), figsize=(12,4), constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.set_title(desc_name[i]);
        im = ax.imshow(model.input_data.descriptor[..., i]);
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal");
        cbar.ax.tick_params();
    @savefig desc_optimize_ann_user_guide.png
    fig.suptitle("Physiographic descriptors");

.. ipython:: python

    # Reset figsize to the Matplotlib default
    plt.figure(figsize=plt.rcParamsDefault['figure.figsize']);

---------------------------
Defining the neural network
---------------------------

If you do not specify the neural network (``net`` argument) in the :meth:`smash.Model.ann_optimize` method, 
a default network will be used to learn the descriptors-to-parameters mapping. 
This example shows how to define a custom neural network using the :meth:`smash.Net` method.

To define a custom neural network, you may need to have information about the physiographic descriptors and hydrological parameters. 
This information will be used to determine the input and output layers of the network, including the number of descriptors, 
the control vector, and the boundary condition (if you want to scale the network output to the boundary condition). 
The default values of these parameters can be obtained as follows:

.. ipython:: python

    problem = model.get_bound_constraints()
    problem

.. ipython:: python

    ncv = problem["num_vars"]  # number of control vector
    cv = problem["names"]  # control vector
    bc = problem["bounds"]  # default boundary condition
    nd = model.input_data.descriptor.shape[-1]  # number of descriptors

Next, we need to initialize the Net object:

.. ipython:: python

    net = smash.Net()

Then, we can define a graph for our custom neural network by specifying the number of layers, type of activation function, 
and output scaling. For example, we can define a neural network with 2 hidden dense layers followed by ``ReLU`` activation functions 
and a final layer followed by a ``sigmoid`` function. To scale the network output to the boundary condition, 
we apply a ``MinMaxScale`` function:

.. ipython:: python

    net.add(layer="dense", options={"input_shape": (nd,), "neurons": 48})
    net.add(layer="activation", options={"name": "ReLU"})
    net.add(layer="dense", options={"neurons": 16})
    net.add(layer="activation", options={"name": "ReLU"})
    net.add(layer="dense", options={"neurons": ncv})
    net.add(layer="activation", options={"name": "sigmoid"})
    net.add(layer="scale", options={"bounds": bc})

Make sure to compile the network after defining it. We can also specify the optimizer and its hyperparameters:

.. ipython:: python

    net.compile(optimizer="Adam", learning_rate=0.004, random_state=23)
    net  # display a summary of the network

---------------------------
Training the neural network
---------------------------

Now, we can train the neural network with the custom graph using the :meth:`smash.Model.ann_optimize` method. 
This method performs operation in-place on ``net``:

.. ipython:: python
        :suppress:
    
        model_ann = model.ann_optimize(
                net, 
                epochs=100, 
                control_vector=cv, 
                bounds=dict(zip(cv, bc))
            )

.. ipython:: python
        :verbatim:
    
        model_ann = model.ann_optimize(
                net, 
                epochs=100, 
                control_vector=cv, 
                bounds=dict(zip(cv, bc))
            )

Some information are also provided during the training process:

.. code-block:: text

    </> ANN Optimize Model
        Mapping: 'ANN' k(x) = N(D1, ..., Dn)
        Optimizer: adam
        Learning rate: 0.004
        Jobs function: [ nse ]
        wJobs: [ 1.0 ]
        Nx: 172
        Np: 52 [ cp cft exc lr ]
        Ns: 0 [  ]
        Ng: 1 [ Y3204040 ]
        wg: 1 [ 1.0 ]

        At epoch      1    J =  1.108795    |proj g| =  0.000060                    
        At epoch      2    J =  1.092934    |proj g| =  0.000062                    
        At epoch      3    J =  1.068635    |proj g| =  0.000062                    
        At epoch      4    J =  1.036452    |proj g| =  0.000060                    
        At epoch      5    J =  0.996259    |proj g| =  0.000057                    
        At epoch      6    J =  0.943647    |proj g| =  0.000051                    
        At epoch      7    J =  0.875171    |proj g| =  0.000067                    
        At epoch      8    J =  0.796573    |proj g| =  0.000132                    
        At epoch      9    J =  0.711945    |proj g| =  0.000190                    
        At epoch     10    J =  0.622793    |proj g| =  0.000198                    
        At epoch     11    J =  0.536326    |proj g| =  0.000148                    
        At epoch     12    J =  0.465874    |proj g| =  0.000075                    
    ...                  
        At epoch     98    J =  0.136344    |proj g| =  0.000023                    
        At epoch     99    J =  0.136195    |proj g| =  0.000023                    
        At epoch    100    J =  0.136045    |proj g| =  0.000020                    
    Training: 100%|███████████████████████████████| 100/100 [00:03<00:00, 26.63it/s]

.. note::

    To ensure the order of the control vectors and to prevent any potential conflicts, it is recommended that you redefine 
    the ``control_vector`` and ``bounds`` arguments in the :meth:`smash.Model.ann_optimize` function as the code above.

------------------------
Visualization of results
------------------------

To visualize the descent of the cost function, we use the ``net`` object and create a plot of the cost function value versus 
the number of iterations. Here's an example:

.. ipython:: python

    y = net.history["loss_train"]
    x = range(1, len(y) + 1)
    plt.plot(x, y);
    plt.xlabel("Epoch");
    plt.ylabel("$1-NSE$");
    plt.grid(alpha=.7, ls="--");
    @savefig cost_optimize_ann_user_guide.png
    plt.title("Cost funciton descent");

.. note::

    By default, ``nse`` is used to define the objective function if you do not specify the ``jobs_fun`` argument 
    in :meth:`smash.Model.ann_optimize`.

The simulated discharge:

.. ipython:: python

    qo = model_ann.input_data.qobs[0,:].copy()
    qo = np.where(qo<0, np.nan, qo)  # to deal with missing data
    plt.plot(qo, label="Observed discharge");
    plt.plot(model_ann.output.qsim[0,:], label="Simulated discharge");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model_ann.mesh.code[0]);
    @savefig qsim_optimize_pre-regio_ann_user_guide.png
    plt.legend();

The cost value:

.. ipython:: python

    model_ann.output.cost

And finally, the spatially distributed model parameters constrained by physiographic descriptors:

.. ipython:: python

    ma = (model_ann.mesh.active_cell == 0)

    ma_cp = np.where(ma, np.nan, model_ann.parameters.cp)
    ma_cft = np.where(ma, np.nan, model_ann.parameters.cft)
    ma_lr = np.where(ma, np.nan, model_ann.parameters.lr)
    ma_exc = np.where(ma, np.nan, model_ann.parameters.exc)
    
    f, ax = plt.subplots(2, 2)
    
    map_cp = ax[0,0].imshow(ma_cp);
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    
    map_cft = ax[0,1].imshow(ma_cft);
    f.colorbar(map_cft, ax=ax[0,1], label="cft (mm)");
    
    map_lr = ax[1,0].imshow(ma_lr);
    f.colorbar(map_lr, ax=ax[1,0], label="lr (min)");
    
    map_exc = ax[1,1].imshow(ma_exc);
    @savefig theta_sd_optimize_pre-regio_ann_user_guide.png
    f.colorbar(map_exc, ax=ax[1,1], label="exc (mm/d)");
