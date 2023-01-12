.. _math_num_description.hydrological_operators:

======================
Hydrological operators
======================


GR like
*******

.. warning::
   to do menus cliquables pr chaque famille de modele (à termes GR, VIC, Lissflood, marine, ...)

Interception
------------

Given an interception storage :math:`\mathcal{I}` of maximum capacity :math:`ci`. If potential evapotranspiration :math:`E` is greater than the sum of liquid precipitation :math:`P`, 			snow melt :math:`mlt` and initial level of the interception storage :math:`h_i`, then the interception reservoir is emptied. Conversely, if the sum of liquid precipitation, snow melt 			and initial level of the interception storage is greater than potential evapotranspiration, the interception storage is filled in depending on it's available storage:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
        &E_i(t)& &=& &\min \left[ E(t), \; P(t) + mlt(t) + h_i(t-1) \right]& \\
        
        &P_{th}(t)& &=& &\max \left[ 0, \; P(t) + mlt(t) + h_i(t-1) - ci - E_i(t) \right]& \\
        
        &h_i(t)& &=& &h_i(t-1) + P(t) + mlt(t) - E_i(t) - P_{th}(t)&
    
    \end{eqnarray}
    
where :math:`P_{th}` corresponds to the remaining rainfall amount (throughfall) inflowing next flow operators.

.. warning::
   
   explain flux matching Ficci
   

Production
----------


Non conservative exchange
-------------------------

Transfer
--------


VIC like
********

HBV like
********

Generic
*******

Surface routing
***************

Linear Reservoir
----------------

Kinematic wave
--------------

Non-linear conceptual - Igamma
------------------------------
   

Sub-surface routing
*******************
