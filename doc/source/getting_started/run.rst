.. _getting_started_run:

===
Run
===

Direct run
----------

.. ipython:: python

	import smash
	import matplotlib.pyplot as plt

	setup, mesh = smash.load_dataset("Cance")

	model = smash.Model(setup, mesh)

	model.adjoint_test('gt', inplace=True);

	plt.plot(model.output.an, abs(1 - model.output.ian));
	plt.xscale("log");
	@savefig plot_gt.png
	plt.yscale("log");	
