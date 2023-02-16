.. _math_num_documentation.cost_functions:

==============
Cost functions
==============

Let :math:`J` be the cost function to minimize and :math:`\theta\in\Omega\subset\mathbb{R}^{N_{p}}` 
be the :math:`N_p`-dimensional model parameters. 
Then, the problem is to find an optimal solution :math:`\hat{\boldsymbol{\theta}}` minimizing the cost function:

.. math ::

	\hat{\boldsymbol{\theta}}=\arg\min_{\boldsymbol{\theta}}J\left(\boldsymbol{\theta}\right).

As classically done in VDA, we assume here that :math:`J` be a convex and differentiable cost function such that:

.. math ::

    J(\theta) = \alpha J_{obs}(\theta) + \beta J_{reg}(\theta)
    
with :math:`J_{obs}` measuring the misfit to observations, :math:`J_{reg}` being a regularization term, :math:`\alpha` the misfit to observations weight and :math:`\beta` the regularization weight.

Firstly, the observation term :math:`J_{obs}` can be generally expressed as follows for multi-gauge observation:

.. math ::
    :nowrap:
    
    \begin{eqnarray}

        &J_{obs}& &=& &\sum_{k=1} ^ {N_o} \phi_k J_{obs,k}& \\
        &J_{obs}& &=& &\text{med}(J_{obs,k})&

    \end{eqnarray}
    
with :math:`J_{obs,k}` and :math:`\phi_{k}` being respectively the constrained term and 
weight associated to each gauge :math:`k\in\left[1..N_{o}\right]`. 
The weighting is such that :math:`\sum_{k=1}^{N_{o}}\phi_{k}=1`. 
Note that :math:`N_{o}=1` is the classical “mono-gauge” downstream calibration case.

Then each term :math:`J_{obs,k}` can be broken down into "classical" objective function (COF) and 
"signatures-based" objective function (SOF) as follows:

.. math ::
    
    J_{obs,k} = w_d j_{k,d} + \sum_{i=1} ^ {N_s} w_{s,i} j_{k,s,i}
    
with :math:`j_{k,d}` being the COF calculated at gauge :math:`k`, 
which is the essential criteria for the optimization, 
:math:`j_{k,s,i}` being the SOF calculating the relative error 
between simulation and observation associated to signature type :math:`i\in\left[1..N_{s}\right]` at gauge :math:`k`, and 
:math:`w_d,w_{s,i}` being the corresponding optimization weight.

COF can be one of:

- ``nse``
    
.. math::
    
    j_{k,d} = j_{k,\text{nse}} = \frac{\sum_{t=t^*} ^ {T} \left[ Q(k,t) - Q_o(k,t) \right] ^ 2}{\sum_{t=t^*} ^ {T} \left[ Q_o(k,t) - \overline{Q_o}(k,t) \right] ^ 2}

- ``kge``

.. math::
    \begin{eqnarray}
        j_{k,d} = \; &j_{k,\text{kge}}& &=& &\sqrt{(r - 1) ^ 2 + (\alpha - 1) ^ 2 + (\beta - 1) ^ 2}& \\ 
        &r& &=& &\frac{\text{Cov} \left[ Q(k), \; Q_o(k) \right]}{\sqrt{\text{Var} \left[Q(k) \right]} \; \sqrt{\text{Var} \left[Q_o(k) \right]}}& \\
        &\alpha& &=& &\frac{\overline{Q(k)}}{\overline{Q_o(k)}}& \\
        &\beta& &=& &\frac{\sqrt{\text{Var}\left[Q(k) \right]}}{\sqrt{\text{Var}\left[Q_o(k) \right]}}
    \end{eqnarray}
    
- ``kge2``

.. math::
    
    j_{k,d} = j_{k, \text{kge2}} = \left( j_{k, \text{kge}} \right) ^ 2
    
- ``se``

.. math::

    j_{k,d} = j_{k, \text{se}} = \sum_{t=t^*} ^ {T} \left[ Q(k,t) - Q_o(k,t) \right] ^ 2
    
- ``rmse``

.. math::

    j_{k,d} = j_{k, \text{rmse}} = \sqrt{\frac{j_{k, \text{se}}}{T - t^* + 1}}
    
- ``logarithmic``

.. math::

    j_{k,d} = j_{k, \text{logarithmic}} = \sum_{t=t^*} ^ {T} \left[ Q_o(k,t) \left( \ln \frac{Q(k,t)}{Q_o(k,t)} \right) ^ 2 \right]

Now, denote :math:`S_{i}^{o}` and :math:`S_{i}^{s}`
are observed and simulated signature type :math:`i` respectively. 
These signatures are defined and calculated as depicted in :ref:`hydrological signatures <math_num_documentation.hydrological_signature>` section. 
Then, for each signature type :math:`i`, the corresponding SOF is computed depending on if the signature is:

- continuous signature:

.. math::

    j_{k,s,i} = \left(\frac{S_{i}^{s}(k)}{S_{i}^{o}(k)}-1\right)^2

- flood event signature:

.. math::

    j_{k,s,i} = \frac{1}{N_E}\sum_{e=1}^{N_{E}}\left(\frac{S_{i,e}^{s}(k)}{S_{i,e}^{o}(k)}-1\right)^2

where :math:`S_{i,e}^{s},S_{i,e}^{o}` are the simulated and observed signature of event number :math:`e\in\left[1..N_{E}\right]`.

.. note::

    The square operators appear in the formulas to ensure the convexity and differentiable property of the cost function when optimizing with VDA algorithms.

Regularization term :math:`J_{reg}` (**TODO**: à compléter)
