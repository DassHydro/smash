.. _math_num_documentation.cost_function:

=============
Cost Function
=============

This section follows on from the :ref:`math_num_documentation.forward_inverse_problem.cost_function` section introduced in the forward problem.
The aim of this section is to present how is constructed the observation :math:`J_{obs}` and regularization :math:`J_{reg}` terms and
how the regularization weighting coefficient :math:`\alpha` can be automatically estimated.

Remember the expression for the cost function :math:`J`

.. math::

    J = J_{obs} + \alpha J_{reg}


Observation term
----------------

First, we can expressed the observation term at a gauge :math:`J_{obs, g}` such as:

.. math::

    J_{obs, g} = \sum_{c=1}^{N_c} w_c j_c \;\;\; \forall g \in [1, N_g]

with :math:`j_c` and :math:`w_c` being respectively any efficiency metric or signature (see sections :ref:`math_num_documentation.efficiency_error_metric`
and :ref:`math_num_documentation.hydrological_signature`) and associated weight.

Then, the observation terms :math:`J_{obs, g}` for each gauge :math:`g` are aggregated to give the final observation term :math:`J_{obs}` such as:

.. math::

    J_{obs} = \sum_{g=1}^{N_g} w_g J_{obs, g}

with :math:`J_{obs, g}` and :math:`w_g` being respectively the observation term and weight associated to each gauge :math:`g\in[1, N_g]`.

Another less standard method of aggregating observation terms at each gauge :math:`J_{obs, g}` is as follows:

.. math::

    J_{obs} = \text{quant}\left(J_{obs, 1}, ..,  J_{obs, N_g}, \; q\right)

with :math:`\text{quant}` the quantile function and :math:`q` the associated quantile value.

.. note::

    :math:`q` can take the following values 0.25, 0.5 and 0.75 which are respectively the lower quartile, the median and the upper quartile.

Regularization term
-------------------

The regularization term :math:`J_{reg}` can be expressed as follows:

.. math::

    J_{reg} = \sum_{c=1}^{N_c} w_c j_c

with :math:`j_c` and :math:`w_c` being respectively any regularization function (see section :ref:`math_num_documentation.regularization_function`)
and associated weight.

.. _math_num_documentation.cost_function.regularization_weighting_coefficient:

Regularization weighting coefficient
------------------------------------

Typically, the value of the weighting coefficient :math:`\alpha` is set by the user according to the weight that is to be placed on 
the observation :math:`J_{obs}` and regularization :math:`J_{reg}` term. Certain methods can be used to determine a more appropriate coefficient.

fast
****

This method consists of a single optimisation iteration to determine the coefficient :math:`\alpha`:

.. math::

    \alpha = \frac{J_{obs, 0} - J_{obs}}{J_{reg}}

with :math:`J_{obs, 0}` the observation term before optimizing and :math:`J_{obs}`, :math:`J_{reg}` respectively the observation and 
regularization term after optimization.

l-curve
*******

This method consists of a series of 6 optimisation iterations to determine the coefficient :math:`\alpha` based on L-Curve:

.. warning::
    Section in development
