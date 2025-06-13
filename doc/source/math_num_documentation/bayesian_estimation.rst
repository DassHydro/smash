.. _math_num_documentation.bayesian_estimation:

===================
Bayesian Estimation
===================

The aim of this section is to present the Bayesian approach currently implemented in `smash` for parameter estimation and uncertainty quantification.

Notation and General Setup
--------------------------

Observed streamflow time series measured at :math:`G` gauged sites during :math:`T` time steps are noted :math:`\boldsymbol{y}^{*}=\left( y^{*}_{g,t} \right)_{g=1:G,t=1:T}`. 
Missing values may be present to account for series having different length at each site or for episodic missing data.

The streamflow time series simulated by the model :math:`\mathcal{M}` at the same sites and time steps are noted :math:`\boldsymbol{y}(\boldsymbol{\rho})=\left( y_{g,t} (\boldsymbol{\rho}) \right)_{g=1:G,t=1:T}`. The vector :math:`\boldsymbol{\rho}` contains the parameters that need to be estimated (typically the parameters of the mapping operator described in section :ref:`math_num_documentation.forward_inverse_problem.mapping`). This notation is the same as the one used in the :ref:`math_num_documentation.forward_inverse_problem` section, except that the dependence on atmospheric forcings :math:`\mathcal{\boldsymbol{I}}`, catchment physiographic descriptors :math:`\boldsymbol{\mathcal{D}}` and intial state values :math:`\boldsymbol{h}_0` has been omitted for simplicity. 

The model :math:`\mathcal{M}` will not be able to perfectly reproduce the observed streamflows, for two main reasons. First, it is, like any model, an imperfect and simplified representation of the real world. Second, the observed streamflows are themselves imperfect estimates of the actual streamflows, due to errors affecting the streamflow observation process (measurements, rating curves, etc.). Consequently, the discrepancy between observed and simulated streamflow results from both :math:`\mathcal{M}` **structural errors** :math:`\delta` and **observation errors** :math:`\varepsilon`. This leads to the following formulation:

.. math::
    :name: math_num_documentation.bayesian_estimation.errors
	
	y^{*}_{g,t} = y_{g,t}(\boldsymbol{\rho}) + \delta_{g,t} + \varepsilon_{g,t}

Both structural errors :math:`\delta_{g,t}` and observation errors  :math:`\varepsilon_{g,t}` are unknown and will therefore be treated as random variables. Note that while both sources of errors seem to play a similar role in this equation, they have a fundamentally different nature. Indeed, observation errors exist independently of the model :math:`\mathcal{M}`, and their properties (e.g. variance) can therefore be evaluated before any model run by means of an uncertainty analysis of the data acquisition process. By contrast, structural errors are by definition associated with the model :math:`\mathcal{M}`, and their properties therefore need to be inferred as part of model estimation, together with the :math:`\mathcal{M}`-parameters :math:`\boldsymbol{\rho}`.

Probabilistic models
--------------------

The first step of Bayesian inference is to hypothesize a **probabilistic model** that could have generated the analyzed data - here, observed streamflows :math:`\boldsymbol{y}^{*}`. Given the formulation in :ref:`equation 1 <math_num_documentation.bayesian_estimation.errors>`, this is equivalent to proposing probabilistic models for errors :math:`\varepsilon_{g,t}` and :math:`\delta_{g,t}` -  the simulated streamflows :math:`y_{g,t}(\boldsymbol{\rho})` being computed by model :math:`\mathcal{M}` in a deterministic way. The following probabilistic modeling assumptions are made in `smash`:

.. math::
    :name: math_num_documentation.bayesian_estimation.error_models
    
	\begin{equation}
	\begin{cases}
	(\varepsilon_{g_1,t_1} \perp \varepsilon_{g_2,t_2}),
	(\delta_{g_1,t_1} \perp \delta_{g_2,t_2}),
	(\delta_{g_1,t_1} \perp \varepsilon_{g_2,t_2}), \forall (g_1,t_1) \neq (g_2,t_2) \\
	\varepsilon_{g,t} \sim  \mathcal{N}(0,u_{g,t}^2) \\
	\delta_{g,t}  \sim \mathcal{N} \left( \phi_{\mu}(y_{g,t},\boldsymbol{\mu}),\phi^2_{\sigma}(y_{g,t},\boldsymbol{\sigma}) \right)
	\end{cases}
	\end{equation}

The first line declares general independence: observation and structural errors are assumed to be independent in space, in time and between the two error types. Spatial correlations, autocorrelations and cross-correlations are not supported in `smash` for the moment. Note however that these independence assumptions apply to *errors* and not to *streamflows*. In other words, the space/time dependence in observed streamflows is not ignored, but rather it is assumed to be fully explained by the space/time dependence in simulated streamflows.  

The second line stipulates that observation errors are realizations from a Gaussian distribution with mean zero and variance :math:`u_{g,t}^2` varying both in time and between gauging stations. The variances :math:`u_{g,t}^2` are assumed known, typically from a preliminary uncertainty analysis performed at each station.

Lastly, the third line stipulates that structural errors are realizations from a Gaussian distribution. The mean and the variance of this distribution may vary with the simulated streamflow :math:`y_{g,t}` according to some mapping functions :math:`\phi_{\mu}` and :math:`\phi_{\sigma}`. Typical choices for the mapping functions are as follows:

* :math:`\phi_{\mu}=0` and :math:`\phi_{\sigma}=\sigma_{0,g}`. In other words, structural errors are assumed to have a zero mean and a site-specific variance :math:`\sigma_{0,g}^2` that does not vary in time. Note that parameters :math:`\boldsymbol{\sigma}= \left( \sigma_{0,g} \right)_{g=1:G}` are unknown and therefore need to be estimated.
* In most case studies, one empirically observes that the variance of structural errors tend to increase with the simulated streamflow. This can be accounted for with the mapping functions :math:`\phi_{\mu}=0` and :math:`\phi_{\sigma}=\sigma_{0,g} + \sigma_{1,g} \times y_{g,t}`. Parameters :math:`\boldsymbol{\sigma}= \left( \sigma_{0,g},\sigma_{1,g} \right)_{g=1:G}` need to be estimated. This combination is the default choice we recommend for most case studies.
* Other mapping functions :math:`\phi_{\sigma}` include: 

1. the power function :math:`\phi_{\sigma}=\sigma_{0,g} + \sigma_{1,g} \times y_{g,t}^{\sigma_{2,g}}` (enabling a non-linear relation with the simulated streamflow);
2. the exponential function :math:`\phi_{\sigma}=\sigma_{0,g} + (\sigma_{2,g}-\sigma_{0,g}) \times \left( 1-\exp (-y_{g,t}/\sigma_{1,g}) \right)` (which introduces an upper bound :math:`\sigma_{2,g}`);
3. the gaussian function :math:`\phi_{\sigma}=\sigma_{0,g} + (\sigma_{2,g}-\sigma_{0,g}) \times \left( 1-\exp(-(y_{g,t}/\sigma_{1,g})^2) \right)` (a variation on the exponential above); 

* A mapping function :math:`\phi_{\mu} \neq 0` may be used to allow structural errors to have a non-zero mean. This might be useful to identify systematic biases in the simulated streamflow. Only two non-zero functions are available at the moment: constant :math:`\phi_{\mu}=\mu_{0,g}` and linear :math:`\phi_{\mu}=\mu_{0,g} + \mu_{1,g} \times y_{g,t}`. We stress that this is a highly experimental feature that has not be thoroughly evaluated yet: we therefore recommend to use it with care and, if in doubt, to stick to the zero-mean mapping :math:`\phi_{\mu}=0`.

Likelihood
----------

Under the assumptions described in the previous section, the likelihood function can be computed as follows:

.. math::
    :name: math_num_documentation.bayesian_estimation.likelihood
    
	\begin{equation}
	p(\boldsymbol{y}^{*} | \boldsymbol{\rho},\boldsymbol{\mu},\boldsymbol{\sigma})=
	\prod_{g=1}^G \prod_{t=1}^T f_{\mathcal{N}} \left( y^{*}_{g,t}; y_{g,t}(\boldsymbol{\rho}) + \phi_{\mu}(y_{g,t},\boldsymbol{\mu}), \phi^2_{\sigma}(y_{g,t},\boldsymbol{\sigma}) + u_{s,t}^2 \right)
	\end{equation}

where :math:`f_{\mathcal{N}} \left( x; m, v \right)` is the `Gaussian probability density function <https://en.wikipedia.org/wiki/Normal_distribution>`_ (pdf) with mean :math:`m` and variance :math:`v` evaluated at :math:`x`. Note that if an observed streamflow is missing, the corresponding term is simply dropped from the double product.

Maximizing the likelihood function in :ref:`equation 3 <math_num_documentation.bayesian_estimation.likelihood>` with respect to the unknown parameters :math:`(\boldsymbol{\rho},\boldsymbol{\mu},\boldsymbol{\sigma})` provides an estimate of these parameters. The likelihood can therefore play a role similar to the cost function described in the :ref:`math_num_documentation.forward_inverse_problem` section. In fact, formal equivalences can even be demonstrated in some cases: for instance, maximizing the likelihood obtained with :math:`\phi_{\mu}=0` and :math:`\phi_{\sigma}=\sigma_{0,g}` is equivalent to minimizing a 'sum of squares' cost function. This will be illustrated in the case studies.


Prior distribution
-------------------

Prior distributions can be specified for all inferred quantities, including the :math:`\mathcal{M}`-parameters :math:`\boldsymbol{\rho}` and the structural error parameters :math:`(\boldsymbol{\mu},\boldsymbol{\sigma})`. Independent priors are used for each individual parameter, leading to the following joint prior pdf: 

.. math::
    :name: math_num_documentation.bayesian_estimation.prior
    
	\begin{equation}
	p(\boldsymbol{\rho},\boldsymbol{\mu},\boldsymbol{\sigma})=
	\prod_{i=1}^{N_{\rho}} p(\rho_i) \prod_{i=1}^{N_{\mu}} p(\mu_i) \prod_{i=1}^{N_{\sigma}} p(\sigma_i) 
	\end{equation}
	
The following distributions are available to specifiy individual priors: `Gaussian <https://en.wikipedia.org/wiki/Normal_distribution>`_, `lognormal <https://en.wikipedia.org/wiki/Log-normal_distribution>`_, `uniform <https://en.wikipedia.org/wiki/Continuous_uniform_distribution>`_, `triangular <https://en.wikipedia.org/wiki/Triangular_distribution>`_, `exponential <https://en.wikipedia.org/wiki/Exponential_distribution>`_ and the improper `flat prior <https://en.wikipedia.org/wiki/Prior_probability#Examples>`_ distribution.

The specification of priors is case-specific and depends on the target parameters, the availability of prior knowledge in the studied region or for hydrological model used within `smash`, etc. For instance, a uniform distribution can be used to specify a feasible range for some parameter; alternatively, lognormal priors are useful to specify order-of-magnitude information for strictly positive parameters; a flat prior is typically used in the absence of any specific knowldge on a parameter; etc. Let us just recall the golden rule of prior specification: the data used in the likelihood function (here, observed streamflows) should **NOT** be used to help specifying a prior distribution.

Posterior distribution
----------------------

The posterior pdf of unknown parameters :math:`(\boldsymbol{\rho},\boldsymbol{\mu},\boldsymbol{\sigma})` can be obtained, up a constant of proportionality, by simply multiplying the likelihood of :ref:`equation 3 <math_num_documentation.bayesian_estimation.likelihood>` and the prior pdf of :ref:`equation 4 <math_num_documentation.bayesian_estimation.prior>`:

.. math::
    :name: math_num_documentation.bayesian_estimation.posterior
    
	\begin{equation}
	p(\boldsymbol{\rho},\boldsymbol{\mu},\boldsymbol{\sigma}|\boldsymbol{y}^{*}) \propto
	p(\boldsymbol{y}^{*} | \boldsymbol{\rho},\boldsymbol{\mu},\boldsymbol{\sigma}) \times
	p(\boldsymbol{\rho},\boldsymbol{\mu},\boldsymbol{\sigma})
	\end{equation}

`smash` maximizes this posterior pdf with respect to the unknown parameters :math:`(\boldsymbol{\rho},\boldsymbol{\mu},\boldsymbol{\sigma})` to estimate them. When flat priors are used, this is equivalent to maximizing the likelihood as discussed earlier.


