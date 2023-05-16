.. _math_num_documentation.forward.hydrological_operators:

======================
Hydrological operators
======================

This section details fluxes and states computation for a given cell :math:`x\in\Omega` also denoted :math:`x \in \mathcal{T}_\Omega` and for a given time step :math:`t\in[1..N_t]` considering a regular temporal grid of time step :math:`\Delta t`. For this cell at time step :math:`t`, we denote by :math:`P(t)` and :math:`E(t)` the local total rainfall and evapotranspiration.

.. _math_num_documentation.forward.hydrological_operators.gr:

GR
**

The operators used here come from GR (Génie Rural) lumped and semi-lumped models of the literature (**TODO** ref).

Interception
------------

Given an interception reservoir :math:`\mathcal{I}` of maximum capacity :math:`c_i`. If potential evapotranspiration :math:`E` is greater than the sum of liquid precipitation :math:`P` and initial water level of the interception reservoir :math:`h_i`, then the interception reservoir is emptied. Conversely, if the sum of liquid precipitation and initial level of the interception reservoir is greater than potential evapotranspiration, the interception reservoir is filled in depending on it's available storage:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
        &E_i(t)& &=& &\min \left[ E(t), \; P(t) + h_i(t-1) \right]& \\
        
        &P_{n}(t)& &=& &\max \left[ 0, \; P(t) + h_i(t-1) - c_i - E_i(t) \right]& \\
        
        &h_i(t)& &=& &h_i(t-1) + P(t) - E_i(t) - P_{n}(t)&
    
    \end{eqnarray}
    
where :math:`P_{n}` corresponds to the remaining rainfall amount (throughfall) inflowing next flow operators.

Production
----------

The production component ensures the role of non linear runoff production function. 

Initially proposed for a minimal complexity description of catchment water balance functioning, based on empirical modeling, the GR model :cite:p:`edijatnoMichel1989` considers a production reservoir :math:`\mathcal{P}` of maximum depth :math:`c_p` and water level :math:`h_p`. The neutralized rainfall and evaporation are respectively denoted :math:`P_{n}` and :math:`E_n`, with :math:`E_n = E - E_i`:
    
.. math::
    :nowrap:
    
    \begin{eqnarray}
    
        dh_p = 
        \begin{cases}
        
            &\left( 1 - \left( \frac{h_p}{c_p} \right) ^ 2 \right) dP_{n} &\text{if} \; P_{n} > 0 \\
            &-\frac{h_p}{c_p} \left(2 - \frac{h_p}{c_p} \right) dE_n &\text{otherwise}
        \end{cases}
    
    
    \end{eqnarray}
    
Assuming a stepwise approximation of the inputs :math:`P_n` and :math:`E_n` the temporal integration of these ordinary differential equations, enabling analytical solutions, gives the infiltrating rainfall :math:`P_s` and the actual evapotranspiration :math:`E_s` from the reservoir store :

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    &P_s(t)& &=& &c_p \left( 1 - \left( \frac{h_p(t - 1)}{c_p} \right) ^ 2 \right) \; \frac{\tanh \left( \frac{P_{n}(t)}{c_p} \right) }{1 + \left( \frac{h_p(t - 1)}{c_p} \right) \; \tanh \left( \frac{P_{n}(t)}{c_p} \right)}& \\
    &E_s(t)& &=& &h_p(t - 1) \left( 2 - \frac{h_p(t - 1)}{c_p} \right) \; \frac{\tanh \left( \frac{E_{n}(t)}{c_p} \right) }{1 + \left( 1 - \frac{h_p(t - 1)}{c_p} \right) \; \tanh \left( \frac{E_{n}(t)}{c_p} \right)}&
    
    \end{eqnarray}
    
:math:`h_p` is the water level of the production reservoir and :math:`P_s` and :math:`E_s` are the amount of water gained or lost over :math:`\Delta t` and used to update :math:`h_p` at time step t:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
       h_p \left( t \right) = h_p(t - 1) + P_s(t) - E_s(t)
    
    \end{eqnarray}
    
Non conservative exchange
-------------------------

A non-conservative exchange function, representing deep percolation or inter-catchment groudwater flow for instance, is expressed following :cite:p:`edijatno91`. Given a power law transfer storage :math:`\mathcal{T} _{ft}` of capacity :math:`c_{ft}` and water level :math:`h_{ft}` the exchange term writes:


.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    F(t) = exc \left( \frac{h_{ft}(t - 1)}{c_{ft}} \right) ^ {7 / 2}
    
    \end{eqnarray}

    
Transfer
--------

Lateral flows within pixels is represented with the following transfer formulations.

.. _math_num_documentation.forward.hydrological_operators.single_plaw_dbranch:

Single power law transfer storage and direct branch
'''''''''''''''''''''''''''''''''''''''''''''''''''

Transfer within pixels can be first represented by spliting the runoff :math:`P_r` into :math:`Q9 = 0.9 P_r` inflowing one branch containing a transfer reservoir :math:`\mathcal{T} _{ft}` of capacity :math:`c_{ft}` and water level :math:`h_{ft}`, and the remaining :math:`Q1 = 0.1 P_r` inflowing a direct branch - i.e. without reservoir. Each transfer branch is also inflowed by the exchange term :math:`F`. 

At the begining of the time step, the level of the power law transfer reservoir :math:`\mathcal{T} _{ft}` is updated as:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    h_{ft} \left(t ^ * \right) = \max \left( \epsilon, \; h_{ft}(t - 1) + 0.9 P_r(t)  + F(t) \right)
    
    \end{eqnarray}
    
With :math:`\epsilon>0`, a fixed small constant.

Next, the outflow discharge from the transfer reservoir writes:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    Q_{ft}(t) = h_{ft} \left( t ^ * \right) - \left[ h_{ft} \left( t ^ * \right) ^ {-4} + c_{ft} ^ {-4} \right] ^ {-1 / 4}
    
    \end{eqnarray}
    
    
The level of the transfer reservoir is updated as:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    h_{ft}(t)=h_{ft} \left( t ^ * \right) - Q_{ft}(t)
    
    \end{eqnarray}
    

In the branch without reservoir the outflow discharge writes:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    Q_d(t) = \max \left[0, \; 0.1 P_r(t) + F(t) \right]
    
    \end{eqnarray}
    
With flux :math:`Q_t` inflowing the routing part equal to:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    Q_t(t) = Q_{ft}(t) + Q_d(t)
    
    \end{eqnarray}        

        
Double power law transfer storages and direct branch
''''''''''''''''''''''''''''''''''''''''''''''''''''

Transfer within pixels is represented similarly as in :ref:`math_num_documentation.forward.hydrological_operators.single_plaw_dbranch` but with a second power law transfer reservoir. 
Again, the runoff :math:`P_r` is splitted into :math:`Q9 = 0.9 P_r` and :math:`Q1 = 0.1 P_r`, the latter :math:`Q1` inflowing a direct branch - i.e. without reservoir.  In the reservoirs branch, the inflow :math:`Q9` is separated a second time into :math:`40 \%` and :math:`60 \%` respectively inflowing two transfer reservoirs  :math:`\mathcal{T} _{ft}` and  :math:`\mathcal{T} _{st}`. 

Again, the exchange term :math:`F` is applied to the direct branch and to the reservoir :math:`\mathcal{T} _{ft}`


.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    &h_{ft} \left(t ^ * \right)& &=& &\max \left( \epsilon, \; h_{ft}(t - 1) + 0.9 \times 0.6 \; P_r(t)  + F(t) \right)& \\
    &h_{st} \left(t ^ * \right)& &=& &\max \left( \epsilon, \; h_{st}(t - 1) + 0.9 \times 0.4 \; P_r(t)  \right)&
    
    \end{eqnarray}
    
Next, the outflow discharges from the reservoirs writes:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    &Q_{ft}\left(t \right)& &=& & h_{ft} \left( t ^ * \right) - \left[ h_{ft} \left( t ^ * \right) ^ {-4} + c_{ft} ^ {-4} \right] ^ {-1 / 4} & \\
    &Q_{st}\left(t \right)& &=& & h_{st} \left( t ^ * \right) - \left[ h_{st} \left( t ^ * \right) ^ {-4} + c_{st} ^ {-4} \right] ^ {-1 / 4}&
    
    \end{eqnarray}
    

In the branch without reservoir:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    Q_d(t) = \max \left[0, \; 0.1 P_r(t) + F(t) \right]
    
    \end{eqnarray}
    
The level of the transfer reservoirs is updated as:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    &h_{ft}(t)& &=& &h_{ft} \left( t ^ * \right) - Q_{ft}(t)& \\
    &h_{st}(t)& &=& &h_{st} \left( t ^ * \right) - Q_{st}(t)&
    
    \end{eqnarray}

With flux :math:`Q_t` inflowing the routing part equal to:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    Q_t(t) = Q_{ft}(t) + Q_{st}(t) + Q_d(t)
    
    \end{eqnarray}

Generic
*******

Surface routing
***************

Surface runoff is conveyed from pixels to pixel to the outlet of the basin, following the drainage plan :math:`\mathcal{D}_{\Omega}`. 
Several routing models of different complexity are available.

Linear Reservoir
----------------

Given :math:`N_{\text{xup}}` upstream cells within :math:`\Omega` flowing into cell :math:`x` as imposed by the flow direction map :math:`\mathcal{D}_{\Omega}`, the upstream runoff is:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    Q_{up}(x,t) = \sum _{k} ^ {N_{\text{xup}}} Q(k, t)
    
    \end{eqnarray}
    
With for a given upstream cell :math:`k`:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    Q(k,t) = h_{lr}(k,t) \left[ 1 - \exp \left( - \frac{\Delta t}{60 \; lr} \right) \right] + Q_{up}(k,t) + Q_t(k,t)
    
    \end{eqnarray}
    
    
Updating the level in the routing storage:

.. math::
    :nowrap:
    
    \begin{eqnarray}
    
    h_{lr}(k,t) = h_{lr}(k,t-1) - Q(k,t)
    
    \end{eqnarray}
