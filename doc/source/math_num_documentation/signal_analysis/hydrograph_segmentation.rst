.. _math_num_documentation.hydrograph_segmentation:

=======================
Hydrograph segmentation
=======================

Segmentation algorithm aims to capture important events occuring over the studied period on each catchment. 
We propose an algorithm for capturing flood events with the aid of the rainfall gradient and rainfall energy.

-----------
Description
-----------

First of all, the peaks of discharge are detected using a simple peak
detection algorithm :cite:p:`marcos_duarte_2021_4599319`. 
We thus consider those exceed the ``peak_quant``-quantile (by default ``peak_quant`` = 0.999) of the observed discharge 
as important events (or flood events). Then the starting date of each event can be determined as the moment
it starts raining dramatically sometime since 72h before the flood event. 
For calculating this, we need to compute the gradient
of the rainfall and then we choose the peaks of rainfall gradient
which exceed its 0.8-quantile. These peaks correspond to the moments
when we have a sharp increase in rainfall. Next, the
rainfall energy is computed as the sum of squares of the rainfall
observed in 24h counted from 1h before the peak of rainfall gradient.
So the starting date is the first moment when the rainfall energy
exceeds 0.2 of the maximal rainfall energy observed in 72h before
the peak discharge (w.r.t. gradient criterion). Finally, we aim
to find an ending date based on the baseflow separation. Effectively, we
compute the difference between the discharge and its baseflow from
the peak of discharge until the end of researched period (lasts ``max_duration`` hours (by default ``max_duration`` = 240) from starting date of event). Then, the ending date is the moment
that the difference between the discharge and its baseflow is minimal
in 48h counted from 1 hour before this moment.

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
    occurring "nearly simultaneously", that means all of these events 
    occur in no more than ``max_duration`` hours: :math:`ed_{u+m}<sd_{u}+` ``max_duration``, then we 
    merge these :math:`m+1` events into a single event :math:`(sd_{u},ed_{u+m})`.
