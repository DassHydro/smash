.. _user_guide.in_depth.bayesian_estimation_approach:

==============================
Bayesian Estimation in `smash`
==============================

The aim of this tutorial is to demonstrate the use of a Bayesian approach for parameter estimation and uncertainty quantification in `smash`.
The principle is to estimate parameters by maximizing the density of the :ref:`posterior distribution<math_num_documentation.bayesian_estimation>`, rather than by maximizing a predefined :ref:`efficiency metric<math_num_documentation.efficiency_error_metric>` such as the Nash-Sutcliffe Efficiency (NSE). The advantages of this approach are the following:

1. it allows using prior information on parameters when available;
2. it explicitly recognizes the uncertainties affecting both the model-simulated discharge (structural uncertainty) and the discharge data used to calibrate the model (observation uncertainty);
3. it naturally handles multi-gauge calibration, with a built-in weighting of the information brought by each gauge. 

This tutorial builds on the :ref:`tutorial on model calibration<user_guide.classical_uses.split_sample_temporal_validation>`: we recommend that you follow it first.

Preliminaries
-------------

We start by importing the modules needed in this tutorial.

.. ipython:: python
	
	import smash
	import matplotlib.pyplot as plt
	import numpy as np

We then load the Cance dataset, which is included in the `smash` module, and initialize a `model` object. 

.. ipython:: python
	
	setup,mesh=smash.factory.load_dataset('cance')
	model0 = smash.Model(setup, mesh)

We finally optimize this model using the standard, non-Bayesian approach described in the :ref:`model calibration tutorial<user_guide.classical_uses.split_sample_temporal_validation>`. Note that in addition to the four model parameters ``cp``, ``ct``, ``kexc`` and ``llr``, we also calibrate the initial state values (``hp`` and ``ht``) so that we work with 'good' initial state values in the remainder of this tutorial. In addition, a single gauge is used for calibration (the most downstream one, with code "V3524010").

.. ipython:: python
	
	calibrated_parameters=["cp","ct","kexc","llr","hp","ht"] 
	calibration_gauges=["V3524010"]
	optimize_options={"parameters": calibrated_parameters}
	cost_options={"gauge": calibration_gauges}
	model0.optimize(optimize_options=optimize_options,cost_options=cost_options)

After optimization completes, it is possible to look at estimated parameters using the code below. The function `smash.optimize_control_info` allows retrieving information on the control vector, in particular the names and values of estimated parameters. Note the use of the option ``"control_tfm":'keep'`` to ensure that no transformation is applied to the control vector. Also note that the composition of the control vector is fairly obvious here because model parameters are spatially uniform (which is the default option). When a more complex :ref:`mapping operator<math_num_documentation.mapping>` is used, the composition of the control vector is more tricky because it is composed of parameters of the mapping operator.

.. ipython:: python
	
	optimize_options={"parameters": calibrated_parameters,"control_tfm":'keep'}
	info=smash.optimize_control_info(model=model0,optimize_options=optimize_options)
	for i in range(len(info['name'])):
		print(f"{info['name'][i]} estimated at : {info['x'][i]:.4f}")


Basic Bayesian Estimation
-------------------------

Bayesian estimation works in a very similar way, with two notable differences:

1. the function `smash.bayesian_optimize` has to be called instead of the function `smash.optimize`; in the same vein, the function `smash.bayesian_optimize_control_info` replaces the function `smash.optimize_control_info`;
2. in addition to the four model parameters (``cp``, ``ct``, ``kexc`` and ``llr``), the list of calibrated parameters includes the parameters ``sg0`` and ``sg1`` which control structural uncertainty (see the documentation on :ref:`Bayesian inference<math_num_documentation.bayesian_estimation>` for details): the standard deviation of structural errors is an affine function of the simulated discharge, ``sg0 + sg1*Qsim``.

.. ipython:: python
	
	calibrated_parameters=["cp","ct","kexc","llr","sg0","sg1"]
	calibration_gauges=["V3524010"]
	optimize_options={"parameters": calibrated_parameters}
	cost_options={"gauge": calibration_gauges}
	model=smash.bayesian_optimize(model=model0,optimize_options=optimize_options,cost_options=cost_options)
	optimize_options={"parameters": calibrated_parameters,"control_tfm":'keep'}
	info=smash.bayesian_optimize_control_info(model=model,optimize_options=optimize_options,cost_options=cost_options)
	for i in range(len(info['name'])):
		print(f"{info['name'][i]} estimated at : {info['x'][i]:.4f}")

Note that the parameter values changed quite a bit compared with the previous non-Bayesian calibration approach: for instance, parameter ``ct`` moved from 227 to 198 mm. This is not surprising since the cost function on which calibration is based changed as well.

The function below generates a plot that compares the observed and the simulated discharge time series. Note how the values of ``sg0`` and ``sg1`` are used to compute the standard deviation of structural errors, which in turn allows deriving a 95% uncertainty interval for the simulated discharge using the `two-sigma rule <https://en.wikipedia.org/wiki/68-95-99.7_rule>`_. 

.. ipython:: python
	
	def doPlot(obs,sim,sg0,sg1,title="Observed vs. simulated discharge",xlim=None):
		if xlim is None:
		    xl=[0,len(sim)-1]
		else:
		    xl=xlim
		serr_stdev=sg0+sg1*sim # standard deviation of structural errors
		lower=sim-2*serr_stdev # 2-sigma rule
		upper=sim+2*serr_stdev # 2-sigma rule
		x=np.arange(xl[0],xl[1])
		plt.fill_between(x=x,y1=lower[x],y2=upper[x],alpha=0.3,facecolor='orange',label="95% uncertainty interval");
		plt.plot(x,sim[x], color='orange',label="Simulated discharge");
		plt.plot(x,obs[x], label="Observed discharge");
		plt.xlabel("Time step");
		plt.ylabel("Discharge ($m^3/s$)");
		plt.legend();
		plt.title(title);
		return(plt)

	igauge=0 # index of the calibration gauge
	obs=model.response_data.q[igauge,]
	sim = model.response.q[igauge,]
	sg0=info['x'][4]
	sg1=info['x'][5]
	@savefig bayesian.png
	doPlot(obs=obs,sim=sim,sg0=sg0,sg1=sg1,xlim=[500,1440])


Using informative priors
------------------------

In the preceding calibration, no prior distributions were specified. In such a case, using `improper flat priors <https://en.wikipedia.org/wiki/Prior_probability#Examples>`_ is defined as the default behavior. Informative priors can be used by specifying, for each element of the control vector, a prior distribution and its parameters. Available prior distributions include: `Gaussian <https://en.wikipedia.org/wiki/Normal_distribution>`_, `LogNormal <https://en.wikipedia.org/wiki/Log-normal_distribution>`_, `Uniform <https://en.wikipedia.org/wiki/Continuous_uniform_distribution>`_, `Triangular <https://en.wikipedia.org/wiki/Triangular_distribution>`_, `Exponential <https://en.wikipedia.org/wiki/Exponential_distribution>`_ and the improper `FlatPrior <https://en.wikipedia.org/wiki/Prior_probability#Examples>`_ distribution. The code below shows an example where the specified prior distributions are rather vague, except the one for parameter ``kexc-0`` which controls a non-conservative water loss or gain.

.. ipython:: python
	
	priors={"cp-0": ["LogNormal", [4.6,0.5]],
		    "ct-0": ["LogNormal", [5.3,0.5]],
		    "kexc-0": ["Gaussian", [0,0.001]], # precise prior, constraining kexc-0 to remain close to zero
		    "llr-0": ["Triangle", [24,1,72]],
		    "sg0-V3524010": ["FlatPrior", []],
		    "sg1-V3524010": ["FlatPrior", []]}

These priors can be passed to the `smash.bayesian_optimize` function as an additional cost_options, as shown below. Note that calibrated parameter vector changed quite a bit compared with the previous calibration. Parameter ``kexc-0`` is close to zero, as expected given the prior constraint. Other parameters compensated by changing values, with no obvious loss of performance visible in the figure.

.. ipython:: python
	
	optimize_options={"parameters": calibrated_parameters}
	cost_options={"gauge": calibration_gauges,"control_prior":priors}
	model=smash.bayesian_optimize(model=model0,optimize_options=optimize_options,cost_options=cost_options)
	optimize_options={"parameters": calibrated_parameters,"control_tfm":'keep'}
	info=smash.bayesian_optimize_control_info(model=model,optimize_options=optimize_options,cost_options=cost_options)
	for i in range(len(info['name'])):
		print(f"{info['name'][i]} estimated at : {info['x'][i]:.4f}")
	igauge=0 # index of the calibration gauge
	obs=model.response_data.q[igauge,]
	sim = model.response.q[igauge,]
	sg0=info['x'][4]
	sg1=info['x'][5]
	@savefig bayesian_priors.png
	doPlot(obs=obs,sim=sim,sg0=sg0,sg1=sg1,xlim=[500,1440])

Using multiple gauges for calibration
-------------------------------------

To use data from the 3 gauges as calibration data,we simply add the gauge IDs to the list of calibration gauges.
Note that we go back to using non-informative priors by not specifying any ``control_prior`` in cost_options.
Also, note that since there are 3 gauges, there are now 3 couples of ``(sg0,sg1)`` values, since structural uncertainty is gauge-specific.
The values estimated for ``(sg0,sg1)`` implicitly define the weighting of each gauge: in a nutshell, gauges with large ``(sg0,sg1)`` values (i.e. with large structural uncertainty) will exert less leverage on the calibration. The most important term is ``sg1``, which represents the part of uncertainty proportional to discharge, and which can hence be interpreted as a standard uncertainty in percent (``sg0`` is comparably negligible, except for near-zero discharge values). In the example below, simulation at the downstream gauge "V3524010" is affected by a ~20% standard uncertainty, while simulation at gauge "V3517010" is affected by a much higher ~36% standard uncertainty.

.. ipython:: python
	
	calibration_gauges=["V3524010", "V3515010", "V3517010"]
	optimize_options={"parameters": calibrated_parameters}
	cost_options={"gauge": calibration_gauges}
	model=smash.bayesian_optimize(model=model0,optimize_options=optimize_options,cost_options=cost_options)
	optimize_options={"parameters": calibrated_parameters,"control_tfm":'keep'}
	info=smash.bayesian_optimize_control_info(model=model,optimize_options=optimize_options,cost_options=cost_options)
	for i in range(len(info['name'])):
		print(f"{info['name'][i]} estimated at : {info['x'][i]:.4f}")

The figure below compares the observed and the simulated discharge time series at gauge "V3517010" and indeed shows a quite poor fit, leading to a rather high uncertainty.

.. ipython:: python

	igauge=2 # index of the calibration gauge
	obs=model.response_data.q[igauge,]
	sim = model.response.q[igauge,]
	sg0=info['x'][6]
	sg1=info['x'][9]
	@savefig bayesian_multigauge.png
	doPlot(obs=obs,sim=sim,sg0=sg0,sg1=sg1,xlim=[500,1440])


Recognizing uncertainty in streamflow data
------------------------------------------

Data uncertainties are stored in ``model.u_response_data.q_stdev``. The values represent standard uncertainties, i.e. the standard deviation of measurement errors, and by default they are set to zero. In plain words, calibration data are assumed to be perfect, which is quite unrealistic.

.. ipython:: python
	
	print(model.u_response_data.q_stdev)

It is possible to recognize the existence of uncertainty in calibration data by specifying nonzero values in ``model.u_response_data.q_stdev``. Note that a standard uncertainty needs to be specified for each time step, because uncertainty may strongly vary through the data range. The example below adopts a simple approach where data uncertainty is assumed to be proportional to the measured value (but in principle, the values should derive from a proper uncertainty analysis of the discharge measurement process). At the first gauge, a moderate ~20% data uncertainty is assumed. The second gauge is assumed to provide very precise data (1% uncertainty), while at the opposite the third gauge is assumed to be very imprecise (~50% data uncertainty). Similar to structural uncertainty, data uncertainty acts on the weighting of the information brought by each gauge: a large data uncertainty will decrease the leverage of the gauge on the calibration problem.

.. ipython:: python
	
	model.u_response_data.q_stdev[0,]=0.2*model.response_data.q[0,]
	model.u_response_data.q_stdev[1,]=0.01*model.response_data.q[1,]
	model.u_response_data.q_stdev[2,]=0.5*model.response_data.q[2,]

Re-calibrating the model with these data uncertainties leads to different optimized parameters: for instance, parameter ``cp`` moved from 127 to 142 mm. The parameters of structural errors also changed quite markedly: for instance, at the third gauge, ``sg1`` decreased from 0.37 to 0.12, resulting in a smaller structural uncertainty as shown in the figure. While possibly surprising at first sight, this result can be explained by the fact that the huge ~50% data uncertainty we specified at this gauge is sufficient to explain most of the mismatch between observed and simulated discharge. In plain words, the poor fit at this gauge is due to poor data, not to a poor model. 

.. ipython:: python
	
	model=smash.bayesian_optimize(model=model,optimize_options=optimize_options,cost_options=cost_options)
	optimize_options={"parameters": calibrated_parameters,"control_tfm":'keep'}
	info=smash.bayesian_optimize_control_info(model=model,optimize_options=optimize_options,cost_options=cost_options)
	for i in range(len(info['name'])):
		print(f"{info['name'][i]} estimated at : {info['x'][i]:.4f}")
	igauge=2 # index of the calibration gauge
	obs=model.response_data.q[igauge,]
	sim = model.response.q[igauge,]
	sg0=info['x'][6]
	sg1=info['x'][9]
	@savefig bayesian_qError.png
	doPlot(obs=obs,sim=sim,sg0=sg0,sg1=sg1,xlim=[500,1440])
