.. _math_num_documentation.hortonian_production:

==============================
Hortonian production reservoir
==============================

Production
''''''''''

In the standart case, the instantaneous production rate is the ratio between the state and the capacity of the reservoir :
:math:`\eta = \left( \frac{h_p}{c_p} \right)^2`. The integration of the complementary of :math:`\eta` the rainfall infiltration :math:`p_s`.

.. math::
    :nowrap:

    \begin{eqnarray}

        &p_s = \int_{t-\Delta t}^{t} (1 - \eta) dt \\
    
    \end{eqnarray}
    
We assume the reservoir receive a rainfall of :math:`p_n` at time :math:`t`, then

.. math::
    :nowrap:
    
    \begin{eqnarray}

        &p_s = & c_p \tanh\left(\frac{p_n}{c_p}\right) \frac{1 - \left( \frac{h_p}{c_p} \right)^2}{1 + \frac{h_p}{c_p} \tanh\left( \frac{p_n}{c_p} \right)} \\
        
    \end{eqnarray}


In the case of fast events with a low permeability, :cite:p:`Astagneau_2022` suggests a modification of the rate : 
:math:`\eta = \left( 1 - \gamma \right) \left( \frac{h_p}{c_p} \right)^2 + \gamma` with :math:`\gamma = 1 - \exp(-p_n \times \alpha_1)`
and :math:`\alpha_1` in :math:`mm` per time unit.

.. math::
    :nowrap:

    \begin{eqnarray}

    &p_s& &=& &\int_{t-\Delta t}^{t} (1 - \eta) dt\\

    && &=& &\int_{t-\Delta t}^{t} \left(1 - (1-\gamma) \left(\frac{h_p}{c_p} \right)^2 \right) dt - \int_{t-\Delta t}^{t} \gamma dt\\
    
    && &=& &\left[ \frac{ c_p }{ \sqrt{1-\gamma} } \tanh \left( \frac{\sqrt{1-\gamma} \  h_p}{c_p} \right) \right]_{t-\Delta t}^t - \gamma \Delta t
    
    \end{eqnarray}


We denote :math:`\lambda := \sqrt{1 - \gamma}`, then

.. math::
    :nowrap:
    
    \begin{eqnarray}

    \tanh \left( \lambda \frac{h_p + p_n}{c_p} \right) - \tanh\left( \lambda \frac{h_p}{c_p} \right) &=& 
    \tanh \left( \lambda \frac{p_n}{c_p} \right) \left(1 - \tanh \left( \lambda \frac{h_p + p_n}{c_p} \right) \tanh \left( \lambda \frac{h_p}{c_p} \right) \right) \\
    &=& \tanh \left( \lambda \frac{p_n}{c_p} \right) \left(1 - \frac{ \tanh \left( \lambda \frac{h_p}{c_p} \right) + \tanh \left( \lambda \frac{p_n}{c_p} \right) } { 1 + \tanh \left( \lambda \frac{h_p}{c_p} \right) \tanh \left( \lambda \frac{p_n}{c_p} \right) } \tanh \left( \lambda \frac{h_p}{c_p} \right) \right) \\
    &\sim& \tanh \left( \lambda \frac{p_n}{c_p} \right) \left(1 - \frac{ \lambda \frac{h_p}{c_p} + \tanh \left( \lambda \frac{p_n}{c_p} \right) } { 1 + \lambda \frac{h_p}{c_p} \tanh \left( \lambda \frac{p_n}{c_p} \right) }  \lambda \frac{h_p}{c_p} \right) \\
    &=& \tanh \left( \lambda \frac{p_n}{c_p} \right) \frac{1 - \left( \lambda \frac{h_p}{c_p} \right)^2}{1 + \lambda \frac{h_p}{c_p} \tanh \left( \lambda \frac{p_n}{c_p} \right)}
    \end{eqnarray}
    
Thus

.. math::
    :nowrap:
    
    \begin{eqnarray}

    p_s &=& \frac{c_p}{\lambda} \tanh \left( \lambda \frac{p_n}{c_p} \right) \frac{1 - \left( \lambda \frac{h_p}{c_p} \right)^2}{1 + \lambda \frac{h_p}{c_p} \tanh \left( \lambda \frac{p_n}{c_p} \right)} - \gamma \Delta t
    \end{eqnarray}


.. note::

    Note that if :math:`\alpha_1 = 0`, we return to the general writting of the instantaneous production rate.
    
    
Transfer
''''''''
The second hypothesis consist in changing the partitioning coefficient to get a faster routing module. 
:cite:p:`Astagneau_2022` suggests to split :math:`p_r` into two branch by the :math:`Q_9` coefficient defined as follow:

.. math::
    :nowrap:

    \begin{eqnarray}

        &p_{rr}& =& (1 - Q_9)(p_r + p_{erc}) + l_{exc}\\
        &p_{rd}& =& Q_9(p_r + p_{erc}) \\
        &Q_9& =& 0.9 \tanh(\alpha_2 p_n)^2 + 0.1
        
    \end{eqnarray}

with :math:`\alpha_2` in :math:`mm` per time unit.


.. note::

    If :math:`\alpha_2 = 0`, we return to the ``gr-4`` writting of the transfer.
    
    
