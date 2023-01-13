.. _math_num_description.hydrological_signature:

=======================
Hydrological signatures
=======================

Several signatures describing and quantifying properties of discharge time series are introduced 
in view to analyze and calibrate hydrological models :cite:p:`westerberg2015uncertainty`.
These signatures permit to describe various aspects of the rainfall-runoff behavior such as: 
flow distribution (based for instance on flow percentiles), 
flow dynamics (based for instance on base-flow separation :cite:p:`nathan1990evaluation,lyne1979stochastic`), 
flow timing, etc.. A so-called continuous signature is a signature that can be computed on the whole studied period.
Flood event signatures on the other hand focus on the behavior of the high flows 
that are observed in the flood events. 
These flood event signatures are calculated via a proposed segmentation algorithm as depicted in :ref:`Hydrograph segmentation <math_num_description.hydrograph_segmentation>`.

Denote :math:`P(t)` and :math:`Q(t)` are the rainfall and runoff at time :math:`t\in\mathbf{U}`, where :math:`\mathbf{U}` is the studied period. 
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
     - :math:`quant(Q(t), 0.02)`
     - mm
   * - Cfp10
     -
     - 0.1-quantile from flow duration curve
     - :math:`quant(Q(t), 0.1)`
     - mm
   * - Cfp50
     -
     - 0.5-quantile from flow duration curve
     - :math:`quant(Q(t), 0.5)`
     - mm
   * - Cfp90
     -
     - 0.9-quantile from flow duration curve
     - :math:`quant(Q(t), 0.9)`
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
     - h
   * - Epf
     - Peak flow
     - Peak runoff in flood event
     - :math:`\max_{t\in\mathbf{E}} Q(t)`
     - mm

Next we are interested in investigating the simulation uncertainty, in term
of signatures, depending on the input parameters of the model. Let us consider
the :math:`m`-parameters set of the model :math:`\theta=(x_{1},...,x_{m})`. 
Then a signature type :math:`i` is represented as :math:`s_{i}=f_i(\theta)`. We are interested at
several Sobol indices called, first-order, second-order, and so forth.
The first- (depending on :math:`x_{1}` for example), second- (depending on :math:`x_{1}` and 
:math:`x_{2}` for example) Sobol indices of :math:`s_{i}` are respectively defined as follows:

.. math ::

    S_{i}^{1}=\frac{\mathbb{\mathbb{V}}[\mathbb{E}[s_{i}|x_{1}]]}{\mathbb{\mathbb{V}}[s_{i}]}

and:

.. math ::

    S_{i}^{2}=\frac{\mathbb{\mathbb{V}}[\mathbb{E}[s_{i}|x_{1},x_{2}]]}{\mathbb{\mathbb{V}}[s_{i}]}.

In such a way, :cite:`azzini2021comparison` proposed a method to estimate
the first- and total-order variance-based sensitivity indices (also
called Sobol indices) on parameter sets of Monte-Carlo simulations
via Saltelli generator :cite:p:`saltelli2002making`.