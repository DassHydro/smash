.. _math_num_documentation.efficiency_metric:

=======================
Efficiency/Error Metric
=======================

The aim of this section is to present all the efficiency/error metrics that can be used as part of the observation term in the inverse problem.

Denote :math:`Q` and :math:`Q^*` the simulated and observed discharge respectively and :math:`t\in[1..T]` a time step.

nse
---

The Nash-Sutcliffe Efficiency

.. math::

    j_{nse} = 1 - \frac{\sum_{t=1}^{T}\left(Q(t) - Q^*(t)\right)^2}{\sum_{t=1}^{T}\left(Q^*(t) - \mu_{Q^*}\right)^2}

with :math:`\mu_{Q^*}` the mean of the observed discharge.

nnse
----

The Normalized Nash-Sutcliffe Efficiency

.. math::

    j_{nnse} = \frac{1}{2 - j_{nse}}

kge
---

The Kling-Gupta Efficiency

.. math::

    j_{kge} = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}

with :math:`r` the Pearson correlation coefficient, :math:`\alpha` the term representing the variability of prediction errors and 
:math:`\beta` the bias term. They are defined as follows:

.. math::
    :nowrap:

    \begin{eqnarray}

        &r& &=& &\frac{\text{cov}(Q, Q^*)}{\sigma_Q \sigma_{Q^*}}\\
        &\alpha& &=& &\frac{\sigma_Q}{\sigma_{Q^*}}\\
        &\beta& &=& &\frac{\mu_Q}{\mu_{Q^*}}

    \end{eqnarray}

with :math:`\text{cov}(Q, Q^*)` the covariance between :math:`Q` and :math:`Q^*`, :math:`\mu_{Q}` and :math:`\mu_{Q^*}` the mean of the simulated and observed discharge respectively and 
:math:`\sigma_{Q}` and :math:`\sigma_{Q^*}` the standard deviation of the simulated and observed discharge respectively.

mae
---

The Mean Absolute Error

.. math::

    j_{mae} = \frac{1}{T} \sum_{t=1}^T \lvert Q(t) - Q^*(t) \rvert

mape
----

The Mean Absolute Percentage Error

.. Can't \left \lvert \right \rvert ...

.. math::

    j_{mape} = \frac{1}{T} \sum_{t=1}^T \lvert \frac{Q(t) - Q^*(t)}{Q^*(t)} \rvert

mse
---

The Mean Squared Error

.. math::

    j_{mse} = \frac{1}{T} \sum_{t=1}^T \left(Q(t) - Q^*(t)\right)^2

rmse
----

The Root Mean Squared Error

.. math::

    j_{rmse} = \sqrt{j_{mse}}

lgrm
----

The Logarithmic Error

.. math::

    j_{lgrm} = \sum_{t=1}^T Q^*(t) \ln\left(\frac{Q(t)}{Q^*(t)}\right)^2
