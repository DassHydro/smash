.. _math_num_documentation.precipitation_partitioning:

==========================
Precipitation Partitioning
==========================

The aim of this section is to present the partitioning of the precipitation into liquid and solid components.

Denote :math:`P` the precipitation, :math:`S` the snow, :math:`T_e` the temperature, :math:`x\in\Omega` a given cell and 
:math:`t\in]0 .. T]` a time step.

First, the precipitation :math:`P` and the snow :math:`S` are summed to give the total precipitation :math:`P_T`

.. math::

    P_T(x, t) = P(x, t) + S(x, t) \;\;\; \forall (x, t) \in \Omega \times ]0 .. T]

Then, the liquid ratio :math:`l_r` splits the total precipitation :math:`P_T` into liquid part (precipitation) :math:`P` and solid part
(snow) :math:`S`.

.. math::
    :nowrap:

    \begin{eqnarray}

        &P(x, t)& &=& &l_r(x, t) \times P_T(x, t)\\
        &S(x, t)& &=& &(1 - l_r(x, t)) \times P_T (x, t) \;\;\; \forall (x, t) \in \Omega \times ]0 .. T]

    \end{eqnarray}

where the liquid ratio :math:`l_r` is derived from a classical parametric S-shaped curve :cite:p:`garavaglia2017impact`.

.. math::

    l_r(x, t) = 1 - \left( 1 + \exp\left( \frac{10}{4}T_e(x, t) - 1\right)\right)^{-1} \;\;\; \forall (x, t) \in \Omega \times ]0 .. T]

.. note::

    - If no snow :math:`S` is provided, the total precipitation :math:`P_T` is simply the precipitation :math:`P` 
      on which a liquid ratio :math:`l_r` will be calculated.
    - If snow :math:`S` is provided, the default liquid ratio :math:`l_r` is overwritten.
 

