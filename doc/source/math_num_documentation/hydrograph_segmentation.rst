.. _math_num_documentation.hydrograph_segmentation:

=======================
Hydrograph Segmentation
=======================

Segmentation algorithm aims to capture important events occuring over the study period on each catchment. 
We propose an algorithm for capturing flood events with the aid of the rainfall gradient and rainfall energy.

-----------
Description
-----------

The discharge peaks are first detected using a simple peak detection algorithm :cite:p:`marcos_duarte_2021_4599319`. 
We consider events that exceed the ``peak_quant``-quantile (we set ``peak_quant`` = 0.995 by default) 
of the observed discharge as important events (or flood events), and events are considered to be distinct 
if they are separated by at least 12h. The starting date of the event is considered to be the moment when 
the rain starts to increase dramatically, which is sometime 72h before the discharge peak. 
To calculate this, we compute the gradient of the rainfall and choose the peaks of rainfall gradient that exceed the 0.8-quantile. 
These peaks correspond to the moments when there is a sharp increase in rainfall. 
Next, the rainfall energy is computed as the sum of squares of the rainfall observed in a 24h-period, 
counted from 1h before the peak of rainfall gradient.
The starting date is the first moment when the rainfall energy exceeds 0.2 of the maximal rainfall energy observed 
in the 72h-period before the discharge peak, based on the gradient criterion. 
Finally, we aim to find the ending date by computing the difference between the discharge and its baseflow from
the discharge peak until the end of study period (which lasts for ``max_duration`` hours 
(we set ``max_duration`` = 240 by default) from the starting date of the event). 
The ending date is the moment when the difference between the discharge and its baseflow is minimal in a 48h-period, 
counted from 1 hour before this moment.

---------
Algorithm
---------

For each catchment, considering 2 time series :math:`(T,Q)` and :math:`(T,P)` where:

- :math:`T=(t_{1},...,t_{n})` is time (by hour),
- :math:`Q=(q_{1},...,q_{n})` is the discharge,
- :math:`P=(p_{1},...,p_{n})` is the rainfall.

Detecting peaks that exceed the ``peak_quant``-quantile of the discharge considered as important events:

:math:`E=(t_{i})_{1\leq i\leq n}` s.t. :math:`q_{i}>\text{quant}(Q,` ``peak_quant``:math:`)`.

For :math:`t_{j}\in E`:

- Determining a starting date based on the "rainfall gradient criterion" and the "rainfall energy criterion":

:math:`RE=(t_{k})_{t_{k}\in(t_{j}-72,t_{j})}` s.t. :math:`\nabla P(t_{k})>\text{quant}(\nabla P([t_{j}-72,t_{j}]), 0.8)`,

:math:`f(t_{x})=||(p_{x}-1,...,p_{x}+23)||_{2}`,

:math:`sd=\min(t_{s})_{t_{s}\in RE}` s.t. :math:`f(t_{s})>0.2||(f(t_{j}-72),...,f(t_{j}))||_{\infty}`.

- Determining an ending date based on discharge baseflow :math:`Qb=\text{Baseflow}(Q)`:

:math:`ed=\arg\min_{t_{e}}\sum_{t=t_{e}-1}^{t_{e}+47}|(Q-Qb)(t)|` s.t. :math:`t_{j} \leq t_e \leq sd+` ``max_duration``.

.. note::
 
    If there exists :math:`m+1` :math:`(m>0)` consecutive events :math:`(sd_{u},ed_{u}),...,(sd_{u+m},ed_{u+m})` 
    occurring "nearly simultaneously", then we merge these :math:`m+1` events into a single event :math:`(sd_{u},ed_{u+m})`. 
    Note that the duration of the merged event may exceed the specified maximum duration of ``max_duration`` hours.
