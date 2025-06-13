.. _math_num_documentation.hydrological_signature:

======================
Hydrological Signature
======================

Several signatures describing and quantifying properties of discharge time series are introduced 
in view to analyze and calibrate hydrological models :cite:p:`westerberg2015uncertainty`.
These signatures permit to describe various aspects of the rainfall-runoff behavior such as: 
flow distribution (based for instance on flow percentiles), 
flow dynamics (based for instance on base-flow separation :cite:p:`nathan1990evaluation,lyne1979stochastic`), 
flow timing, etc.. A so-called continuous signature is a signature that can be computed on the whole study period.
Flood event signatures on the other hand focus on the behavior of the high flows 
that are observed in the flood events. 
These flood event signatures are calculated via a proposed segmentation algorithm as depicted in :ref:`Hydrograph segmentation <math_num_documentation.hydrograph_segmentation>`.

Denote :math:`P(t)` and :math:`Q(t)` are the rainfall and runoff at time :math:`t\in\mathbf{U}`, where :math:`\mathbf{U}` is the study period. 
Then :math:`Qb(t)` and :math:`Qq(t)` are the baseflow and quickflow computed using a classical technique for streamflow separation 
(please refer to :cite:p:`lyne1979stochastic` and :cite:p:`nathan1990evaluation` for more details). 
Considering a flood event in a period denoted :math:`\mathbf{E} \subset \mathbf{U}`, 
so all studied continuous signatures (denoted by the letter ``C``) and flood event signatures (denoted by the letter ``E``) 
are given in the table below.

.. list-table:: List of all studied signatures
   :widths: 10 20 50 15 5
   :header-rows: 1

   * - Notation
     - Signature
     - Description
     - Formula
     - Unit
   * - Crc
     - Continuous runoff coefficients
     - Coefficient relating the amount of runoff to the amount of precipitation received
     - :math:`\frac{\int^{t\in\mathbf{U}} Q(t)dt}{\int^{t\in\mathbf{U}} P(t)dt}`
     - --
   * - Crchf
     - 
     - Coefficient relating the amount of high-flow to the amount of precipitation received
     - :math:`\frac{\int^{t\in\mathbf{U}} Qq(t)dt}{\int^{t\in\mathbf{U}} P(t)dt}`
     - --
   * - Crclf
     - 
     - Coefficient relating the amount of low-flow to the amount of precipitation received
     - :math:`\frac{\int^{t\in\mathbf{U}} Qb(t)dt}{\int^{t\in\mathbf{U}} P(t)dt}`
     - --
   * - Crch2r
     - 
     - Coefficient relating the amount of high-flow to the amount of runoff
     - :math:`\frac{\int^{t\in\mathbf{U}} Qq(t)dt}{\int^{t\in\mathbf{U}} Q(t)dt}`
     - --
   * - Cfp2
     - Flow percentiles
     - 0.02-quantile from flow duration curve
     - :math:`\text{quant}(Q(t), 0.02)`
     - mm
   * - Cfp10
     -
     - 0.1-quantile from flow duration curve
     - :math:`\text{quant}(Q(t), 0.1)`
     - mm
   * - Cfp50
     -
     - 0.5-quantile from flow duration curve
     - :math:`\text{quant}(Q(t), 0.5)`
     - mm
   * - Cfp90
     -
     - 0.9-quantile from flow duration curve
     - :math:`\text{quant}(Q(t), 0.9)`
     - mm
   * - Eff
     - Flood flow
     - Amount of quickflow in flood event
     - :math:`\int^{t\in\mathbf{E}} Qq(t)dt`
     - mm
   * - Ebf
     - Base flow
     - Amount of baseflow in flood event
     - :math:`\int^{t\in\mathbf{E}} Qb(t)dt`
     - mm
   * - Erc
     - Flood event runoff coefficients
     - Coefficient relating the amount of runoff to the amount of precipitation received
     - :math:`\frac{\int^{t\in\mathbf{E}} Q(t)dt}{\int^{t\in\mathbf{E}} P(t)dt}`
     - --
   * - Erchf
     - 
     - Coefficient relating the amount of high-flow to the amount of precipitation received
     - :math:`\frac{\int^{t\in\mathbf{E}} Qq(t)dt}{\int^{t\in\mathbf{E}} P(t)dt}`
     - --
   * - Erclf
     - 
     - Coefficient relating the amount of low-flow to the amount of precipitation received
     - :math:`\frac{\int^{t\in\mathbf{E}} Qb(t)dt}{\int^{t\in\mathbf{E}} P(t)dt}`
     - --
   * - Erch2r
     - 
     - Coefficient relating the amount of high-flow to the amount of runoff
     - :math:`\frac{\int^{t\in\mathbf{E}} Qq(t)dt}{\int^{t\in\mathbf{E}} Q(t)dt}`
     - --
   * - Elt
     - Lag time
     - Difference time between the peak runoff and the peak rainfall
     - :math:`\arg\max_{t\in\mathbf{E}} Q(t)` :math:`-\arg\max_{t\in\mathbf{E}} P(t)`
     - dt
   * - Epf
     - Peak flow
     - Peak runoff in flood event
     - :math:`\max_{t\in\mathbf{E}} Q(t)`
     - mm

where :math:`dt` is the time step.

Now, denote :math:`S_s^*` and :math:`S_s` are observed and simulated signature type respectively. For each signature type :math:`s`,
the corresponding signature based efficiency metric is computed depending on if the signature is:

- continuous signature:

.. math::

    j_s = \left(\frac{S_{s}}{S_{s}^{*}}-1\right)^2

- flood event signature:

.. math::

    j_{s} = \frac{1}{N_E}\sum_{e=1}^{N_{E}}\left(\frac{S_{s_e}}{S_{s_e}^{*}}-1\right)^2

where :math:`S_{s_e},S_{s_e}^{*}` are the simulated and observed signature of event number :math:`e\in\left[1..N_{E}\right]`.


