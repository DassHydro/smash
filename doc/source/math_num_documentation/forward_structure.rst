.. _math_num_documentation.forward_structure:

=================
Forward Structure
=================

In `smash`, a forward/direct spatially distributed model is obtained by chaining **differentiable hydrological-hydraulic operators** via simulated fluxes:

- (optional) a descriptors-to-parameters mapping :math:`\phi`, either for parameters imposing spatial constraints and/or regional mapping between physical descriptors and model conceptual parameters, see :ref:`mapping section <math_num_documentation.mapping>`.
- (optional) a ``snow`` operator :math:`\mathcal{M}_{snw}` generating a melt flux :math:`m_{lt}`, which is then summed with the precipitation flux to feed the ``hydrological`` operator :math:`\mathcal{M}_{rr}`.
- A ``hydrological`` production operator :math:`\mathcal{M}_{rr}` generating an elementary discharge :math:`q_t`, which feeds the routing operator. 
- A ``routing`` operator :math:`\mathcal{M}_{hy}` simulating the propagation of discharge :math:`Q`.

The operators' chaining principle is presented in section :ref:`forward and inverse problems statement <math_num_documentation.forward_inverse_problem.chaining>` (cf. Eq. :eq:`math_num_documentation.forward_inverse_problem.forward_problem_Mhy_circ_Mrr`), and the chaining fluxes are explicated in the diagram below. The forward model obtained reads :math:`\mathcal{M}=\mathcal{M}_{hy}\left(\,.\,,\mathcal{M}_{rr}\left(\,.\,,\mathcal{M}_{snw}\left(.\right)\right)\right)`.

This section describes the various operators available in `smash` with mathematical/numerical expressions, **input data** :math:`\left[\boldsymbol{I},\boldsymbol{D}\right](x,t)`, **tunable conceptual parameters** :math:`\boldsymbol{\theta}(x,t)`, and simulated **state and fluxes** :math:`\boldsymbol{U}(x,t)=\left[Q,\boldsymbol{h},\boldsymbol{q}\right](x,t)`.

These operators are written below for a given pixel :math:`x` of the 2D spatial domain :math:`\Omega` and for a time :math:`t` in the simulation window :math:`\left]0,T\right]`.

.. figure:: ../_static/forward_flowchart_detail_input_params_states_fluxes.png
    :align: center
    :width: 800
    
    Diagram of input data, hydrological-hydraulic operators, simulated quantities of a forward model
    :math:`\mathcal{M}=\mathcal{M}_{hy}\left(\,.\,,\mathcal{M}_{rr}\left(\,.\,,\mathcal{M}_{snw}\left(.\right)\right)\right)` (cf. Eq. :eq:`math_num_documentation.forward_inverse_problem.forward_problem_Mhy_circ_Mrr`);
    recall the  composition principle is explained in section :ref:`forward and inverse problems statement <math_num_documentation.forward_inverse_problem>`.
    

.. _math_num_documentation.forward_structure.snow_module:

Snow operator :math:`\mathcal{M}_{snw}`
---------------------------------------

.. image:: ../_static/snow_module.svg
    :align: center
    :width: 300

.. dropdown:: zero (Zero Snow)
    :animate: fade-in-slide-down

    This snow operator simply means that there is no snow operator.

    .. math::
        
        m_{lt}(x, t) = 0

    with :math:`m_{lt}` the melt flux.

.. dropdown:: ssn (Simple Snow)
    :animate: fade-in-slide-down

    This snow operator is a simple degree-day snow operator. It can be expressed as follows:

    .. math::

        m_{lt}(x, t) = f\left(\left[S, T_e\right](x, t), k_{mlt}(x), h_s(x, t)\right)

    with :math:`m_{lt}` the melt flux, :math:`S` the snow, :math:`T_e` the temperature, :math:`k_{mlt}` the melt coefficient and :math:`h_s` the state of the snow reservoir.

    .. note::

        Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
        
        - Internal fluxes: :math:`\{m_{lt}\}\in\boldsymbol{q}`
        - Atmospheric forcings: :math:`\{S, T_e\}\in\boldsymbol{\mathcal{I}}`
        - Parameters: :math:`\{k_{mlt}\}\in\boldsymbol{\theta}`
        - Normalized states: :math:`\{\tilde{h_s}\}`, where :math:`\tilde{h_s}=\frac{h_s}{k_{mlt}}`, with states :math:`\{h_s\}\in\boldsymbol{h}`

    The function :math:`f` is resolved numerically as follows:

    - Update the normalized snow reservoir state :math:`\tilde{h_s}` for :math:`t^* \in \left] t-1 , t\right[`

    .. math::

        \tilde{h_s}(x, t^*) = \tilde{h_s}(x, t-1) + S(x, t)

    - Compute the melt flux :math:`m_{lt}`

    .. math::
        :nowrap:

        \begin{eqnarray}

            m_{lt}(x, t) =
            \begin{cases}

                0 &\text{if} \; T_e(x, t) \leq 0 \\
                \min\left(\tilde{h_s}(x, t^*), k_{mlt}(x)\times T_e(x, t)\right) &\text{otherwise}

            \end{cases}

        \end{eqnarray}

    - Update the normalized snow reservoir state :math:`\tilde{h_s}`

    .. math::

        \tilde{h_s}(x, t) = \tilde{h_s}(x, t^*) - m_{lt}(x, t)

.. _math_num_documentation.forward_structure.hydrological_module:

Hydrological operator :math:`\mathcal{M}_{rr}`
----------------------------------------------

Hydrological processes can be described at pixel scale in `smash` with one of the available hydrological operators adapted from state-of-the-art lumped or distributed models.

.. image:: ../_static/hydrological_module.svg
    :align: center
    :width: 500

.. _math_num_documentation.forward_structure.hydrological_module.gr4:

.. dropdown:: Génie Rural with 4 parameters (gr4)
    :animate: fade-in-slide-down

    This hydrological operator is derived from the GR4 model :cite:p:`perrin2003improvement`.

    .. hint::

        Helpful links about GR:

        - `Brief history of GR models <https://webgr.inrae.fr/models/a-brief-history/>`__
        - `Scientific papers <https://webgr.inrae.fr/publications/articles/>`__
        - `GR models in a R package <https://hydrogr.github.io/airGR/>`__

    .. figure:: ../_static/gr4_structure.svg
        :align: center
        :width: 400
        
        Diagram of the ``gr4`` like hydrological operator

    It can be expressed as follows:

    .. math::

        q_{t}(x, t) = f\left(\left[P, E\right](x, t), m_{lt}(x, t), \left[c_i, c_p, c_t, k_{exc}\right](x), \left[h_i, h_p, h_t\right](x, t)\right)

    with :math:`q_{t}` the elemental discharge, :math:`P` the precipitation, :math:`E` the potential evapotranspiration,
    :math:`m_{lt}` the melt flux from the snow operator, :math:`c_i` the maximum capacity of the interception reservoir,
    :math:`c_p` the maximum capacity of the production reservoir, :math:`c_t` the maximum capacity of the transfer reservoir,
    :math:`k_{exc}` the exchange coefficient, :math:`h_i` the state of the interception reservoir, :math:`h_p` the state of the production reservoir
    and :math:`h_t` the state of the transfer reservoir.

    .. note::

        Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
        
        - Internal fluxes: :math:`\{q_{t}, m_{lt}\}\in\boldsymbol{q}`
        - Atmospheric forcings: :math:`\{P, E\}\in\boldsymbol{\mathcal{I}}`
        - Parameters: :math:`\{c_i, c_p, c_t, k_{exc}\}\in\boldsymbol{\theta}`
        - Normalized states: :math:`\{\tilde{h_i}, \tilde{h_p}, \tilde{h_t}\}`, where :math:`\tilde{h_i} = \frac{h_i}{c_i}`, :math:`\tilde{h_p} = \frac{h_p}{c_p}`, and :math:`\tilde{h_t} = \frac{h_t}{c_t}`, with states :math:`\{h_i, h_p, h_t\} \in \boldsymbol{h}`

    The function :math:`f` is resolved numerically as follows:

    **Interception**

    - Compute interception evapotranspiration :math:`e_i`

    .. math::

        e_i(x, t) = \min(E(x, t), P(x, t) + m_{lt}(x, t) + \tilde{h_i}(x, t - 1)\times c_i(x))

    - Compute the neutralized precipitation :math:`p_n` and evapotranspiration :math:`e_n`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &p_n(x, t)& &=& &\max \left(0, \; P(x, t) + m_{lt}(x, t) - c_i(x) \times (1 - \tilde{h_i}(x, t - 1)) - e_i(x, t) \right)\\

            &e_n(x, t)& &=& &E(x, t) - e_i(x, t)

        \end{eqnarray}

    - Update the interception reservoir state :math:`\tilde{h_i}`

    .. math::

        \tilde{h_i}(x, t) = \tilde{h_i}(x, t - 1) + \frac{P(x, t) + m_{lt}(x, t) + e_i(x, t) - p_n(x, t)}{c_i(x)}

    **Production**

    - Compute the production infiltrating precipitation :math:`p_s` and evapotranspiration :math:`e_s`

    .. math::
        :nowrap:

        \begin{eqnarray}

        &p_s(x, t)& &=& &c_p(x) (1 - \tilde{h_p}(x, t - 1)^2) \frac{\tanh\left(\frac{p_n(x, t)}{c_p(x)}\right)}{1 + \tilde{h_p}(x, t - 1) \tanh\left(\frac{p_n(x, t)}{c_p(x)}\right)}\\

        &e_s(x, t)& &=& &\tilde{h_p}(x, t - 1) c_p(x) (2 - \tilde{h_p}(x, t - 1)) \frac{\tanh\left(\frac{e_n(x, t)}{c_p(x)}\right)}{1 + (1 - \tilde{h_p}(x, t - 1)) \tanh\left(\frac{e_n(x, t)}{c_p(x)}\right)}
        \end{eqnarray}

    - Update the normalized production reservoir state :math:`\tilde{h_p}`

    .. math::

        \tilde{h_p}(x, t^*) = \tilde{h_p}(x, t - 1) + \frac{p_s(x, t) - e_s(x, t)}{c_p(x)}

    - Compute the production runoff :math:`p_r`

    .. math::
        :nowrap:

        \begin{eqnarray}

            p_r(x, t) =
            \begin{cases}

                0 &\text{if} \; p_n(x, t) \leq 0 \\
                p_n(x, t) - (\tilde{h_p}(x, t^*) - \tilde{h_p}(x, t - 1))c_p(x) &\text{otherwise}

            \end{cases}

        \end{eqnarray}

    - Compute the production percolation :math:`p_{erc}`

    .. math::

        p_{erc}(x, t) = \tilde{h_p}(x, t^*) c_p(x) \left(1 - \left(1 + \left(\frac{4}{9}\tilde{h_p}(x, t^*)\right)^4\right)^{-1/4}\right)

    - Update the normalized production reservoir state :math:`\tilde{h_p}`

    .. math::

        \tilde{h_p}(x, t) = \tilde{h_p}(x, t^*) - \frac{p_{erc}(x, t)}{c_p(x)}

    **Exchange**

    - Compute the exchange flux :math:`l_{exc}`

    .. math::

        l_{exc}(x, t) = k_{exc}(x) \tilde{h_t}(x, t - 1)^{7/2}

    **Transfer**

    - Split the production runoff :math:`p_r` into two branches (transfer and direct), :math:`p_{rr}` and :math:`p_{rd}`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &p_{rr}(x, t)& &=& &0.9(p_r(x, t) + p_{erc}(x, t)) + l_{exc}(x, t)\\
            &p_{rd}(x, t)& &=& &0.1(p_r(x, t) + p_{erc}(x, t))

        \end{eqnarray}

    - Update the normalized transfer reservoir state :math:`\tilde{h_t}`

    .. math::
        
        \tilde{h_t}(x, t^*) = \max\left(0, \tilde{h_t}(x, t - 1) + \frac{p_{rr}(x, t)}{c_t(x)}\right)

    - Compute the transfer branch elemental discharge :math:`q_r`

    .. math::
        :nowrap:

        \begin{eqnarray}

            q_r(x, t) = \tilde{h_t}(x, t^*)c_t(x) - \left(\left(\tilde{h_t}(x, t^*)c_t(x)\right)^{-4} + c_t(x)^{-4}\right)^{-1/4}

        \end{eqnarray}

    - Update the normalized transfer reservoir state :math:`\tilde{h_t}`

    .. math::

        \tilde{h_t}(x, t) = \tilde{h_t}(x, t^*) - \frac{q_r(x, t)}{c_t(x)}

    - Compute the direct branch elemental discharge :math:`q_d`

    .. math::

        q_d(x, t) = \max(0, p_{rd}(x, t) + l_{exc}(x, t))

    - Compute the elemental discharge :math:`q_t`

    .. math::

        q_t(x, t) = q_r(x, t) + q_d(x, t)

.. _math_num_documentation.forward_structure.hydrological_module.gr5:

.. dropdown:: Génie Rural with 5 parameters (gr5)
    :animate: fade-in-slide-down

    This hydrological operator is derived from the GR5 model :cite:p:`LeMoine_2008`. It consists in a gr4 like model structure (see diagram above)  with a modified exchange flux with two parameters to account for seasonal variations.

    .. hint::

        Helpful links about GR:

        - `Brief history of GR models <https://webgr.inrae.fr/models/a-brief-history/>`__
        - `Scientific papers <https://webgr.inrae.fr/publications/articles/>`__
        - `GR models in a R package <https://hydrogr.github.io/airGR/>`__

    .. figure:: ../_static/gr5_structure.svg
        :align: center
        :width: 400
        
        Diagram of the ``gr5`` like hydrological operator

    It can be expressed as follows:

    .. math::

        q_{t}(x, t) = f\left(\left[P, E\right](x, t), m_{lt}(x, t), \left[c_i, c_p, c_t, k_{exc}, a_{exc}\right](x), \left[h_i, h_p, h_t\right](x, t)\right)

    with :math:`q_{t}` the elemental discharge, :math:`P` the precipitation, :math:`E` the potential evapotranspiration,
    :math:`m_{lt}` the melt flux from the snow operator, :math:`c_i` the maximum capacity of the interception reservoir,
    :math:`c_p` the maximum capacity of the production reservoir, :math:`c_t` the maximum capacity of the transfer reservoir,
    :math:`k_{exc}` the exchange coefficient, :math:`a_{exc}` the exchange threshold, :math:`h_i` the state of the interception reservoir, 
    :math:`h_p` the state of the production reservoir and :math:`h_t` the state of the transfer reservoir.

    .. note::

        Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
        
        - Internal fluxes: :math:`\{q_{t}, m_{lt}\}\in\boldsymbol{q}`
        - Atmospheric forcings: :math:`\{P, E\}\in\boldsymbol{\mathcal{I}}`
        - Parameters: :math:`\{c_i, c_p, c_t, k_{exc}, a_{exc}\}\in\boldsymbol{\theta}`
        - Normalized states: :math:`\{\tilde{h_i}, \tilde{h_p}, \tilde{h_t}\}`, where :math:`\tilde{h_i} = \frac{h_i}{c_i}`, :math:`\tilde{h_p} = \frac{h_p}{c_p}`, and :math:`\tilde{h_t} = \frac{h_t}{c_t}`, with states :math:`\{h_i, h_p, h_t\} \in \boldsymbol{h}`

    The function :math:`f` is resolved numerically as follows:

    **Interception**

    Same as ``gr4`` interception, see :ref:`GR4 Interception <math_num_documentation.forward_structure.hydrological_module.gr4>`.

    **Production**

    Same as ``gr4`` production, see :ref:`GR4 Production <math_num_documentation.forward_structure.hydrological_module.gr4>`.

    **Exchange**

    - Compute the exchange flux :math:`l_{exc}`

    .. math::

        l_{exc}(x, t) = k_{exc}(x) \left(\tilde{h_t}(x, t - 1) - a_{exc}(x)\right)

    **Transfer**

    Same as ``gr4`` transfer, see :ref:`GR4 Transfer <math_num_documentation.forward_structure.hydrological_module.gr4>`.

.. _math_num_documentation.forward_structure.hydrological_module.gr6:

.. dropdown:: Génie Rural with 6 parameters (gr6)
    :animate: fade-in-slide-down

    This hydrological module is derived from the GR6 model :cite:p:`michel2003, pushpalatha`.

    .. hint::

        Helpful links about GR:

        - `Brief history of GR models <https://webgr.inrae.fr/models/a-brief-history/>`__
        - `Scientific papers <https://webgr.inrae.fr/publications/articles/>`__
        - `GR models in a R package <https://hydrogr.github.io/airGR/>`__

    .. figure:: ../_static/gr6_structure.svg
        :align: center
        :width: 400
        
        Diagram of the ``gr6`` like hydrological operator

    It can be expressed as follows:

    .. math::

        q_{t}(x, t) = f\left(\left[P, E\right](x, t), m_{lt}(x, t), \left[c_i, c_p, c_t, b_e, k_{exc}, a_{exc}\right](x), \left[h_i, h_p, h_t, h_e\right](x, t)\right)

    with :math:`q_{t}` the elemental discharge, :math:`P` the precipitation, :math:`E` the potential evapotranspiration,
    :math:`m_{lt}` the melt flux from the snow module, :math:`c_i` the maximum capacity of the interception reservoir,
    :math:`c_p` the maximum capacity of the production reservoir, :math:`c_t` the maximum capacity of the transfer reservoir,
    :math:`b_e` controls the slope of the recession, 
    :math:`k_{exc}` the exchange coefficient, :math:`a_{exc}` the exchange threshold, :math:`h_i` the state of the interception reservoir, 
    :math:`h_p` the state of the production reservoir and :math:`h_t` the state of the transfer reservoir,
    :math:`h_e` the state of the exponential reservoir.

    .. note::

        Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
        
        - Internal fluxes: :math:`\{q_{t}, m_{lt}\}\in\boldsymbol{q}`
        - Atmospheric forcings: :math:`\{P, E\}\in\boldsymbol{\mathcal{I}}`
        - Parameters: :math:`\{c_i, c_p, c_t, b_e, k_{exc}, a_{exc}\}\in\boldsymbol{\theta}`
        - Normalized states: :math:`\{\tilde{h_i}, \tilde{h_p}, \tilde{h_t}, \tilde{h_e}\}`, where :math:`\tilde{h_i} = \frac{h_i}{c_i}`, :math:`\tilde{h_p} = \frac{h_p}{c_p}`, :math:`\tilde{h_t} = \frac{h_t}{c_t}`, and :math:`\tilde{h_e} = \frac{h_e}{b_e}`, with states :math:`\{h_i, h_p, h_t, h_e\} \in \boldsymbol{h}`

    The function :math:`f` is resolved numerically as follows:


    **Interception**

    Same as ``gr4`` interception, see :ref:`GR4 Interception <math_num_documentation.forward_structure.hydrological_module.gr4>`.

    **Production**

    Same as ``gr4`` production, see :ref:`GR4 Production <math_num_documentation.forward_structure.hydrological_module.gr4>`.

    **Exchange**

    Same as ``gr5`` exchange, see :ref:`GR5 Exchange <math_num_documentation.forward_structure.hydrological_module.gr5>`.

    **Transfer**

    - Split the production runoff :math:`p_r` into three branches (transfer, exponential and direct), :math:`p_{rr}`, :math:`p_{re}` and :math:`p_{rd}`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &p_{rr}(x, t)& &=& &0.6 \times 0.9(p_r(x, t) + p_{erc}(x, t)) + l_{exc}(x, t)\\
            &p_{re}(x, t)& &=& &0.4 \times 0.9(p_r(x, t) + p_{erc}(x, t)) + l_{exc}(x, t)\\
            &p_{rd}(x, t)& &=& &0.1(p_r(x, t) + p_{erc}(x, t))

        \end{eqnarray}

    - Update the normalized transfer reservoir state :math:`\tilde{h_t}`

    .. math::
        
        \tilde{h_t}(x, t^*) = \max\left(0, \tilde{h_t}(x, t - 1) + \frac{p_{rr}(x, t)}{c_t(x)}\right)

    - Compute the transfer branch elemental discharge :math:`q_r`

    .. math::
        :nowrap:

        \begin{eqnarray}

            q_r(x, t) = \tilde{h_t}(x, t^*)c_t(x) - \left(\left(\tilde{h_t}(x, t^*)c_t(x)\right)^{-4} + c_t(x)^{-4}\right)^{-1/4}

        \end{eqnarray}

    - Update the normalized transfer reservoir state :math:`\tilde{h_t}`

    .. math::

        \tilde{h_t}(x, t) = \tilde{h_t}(x, t^*) - \frac{q_r(x, t)}{c_t(x)}


    - Update the normalized exponential state :math:`\tilde{h_e}`

    .. math::
        
        \tilde{h_e}(x, t^*) = \tilde{h_e}(x, t - 1) + p_{re}

    - Compute the exponential branch elemental discharge :math:`q_{e}`

    .. math::
        :nowrap:

        \begin{eqnarray}

            q_{e}(x, t) =
            \begin{cases}
                
                b_e(x) \ln \left( 1 + \exp \left( \frac{\tilde{h_e}(x, t^*)}{b_e(x)} \right) \right) &\text{if} \; -7 \lt \frac{\tilde{h_e}(x, t^*)}{b_e(x)} \lt 7 \\

                b_e(x) * \exp \left( \frac{\tilde{h_e}(x, t^*)}{b_e(x)} \right) &\text{if} \; \frac{\tilde{h_e}(x, t^*)}{b_e(x)} \lt -7 \\

                \tilde{h_e}(x, t^*) + \frac{ b_e(x) }{ \exp \left( \frac{\tilde{h_e}(x, t^*)}{b_e(x)} \right) } \; &\text{otherwise}.

            \end{cases}

        \end{eqnarray}

    - Update the normalized exponential reservoir state :math:`\tilde{h_e}`

    .. math::

        \tilde{h_e}(x, t) = \tilde{h_e}(x, t^*) - q_{e}


    - Compute the direct branch elemental discharge :math:`q_d`

    .. math::

        q_d(x, t) = \max(0, p_{rd}(x, t) + l_{exc}(x, t))

    - Compute the elemental discharge :math:`q_t`

    .. math::

        q_t(x, t) = q_r(x, t) + q_{e}(x, t) + q_d(x, t)

.. _math_num_documentation.forward_structure.hydrological_module.grc:

.. dropdown:: Génie Rural C (grc)
    :animate: fade-in-slide-down

    This hydrological operator is derived from the GR models. It consists in a ``gr4`` like model structure
    with a second transfer reservoir.

    .. figure:: ../_static/grc_structure.svg
        :align: center
        :width: 300
        
        Diagram of the ``grc`` hydrological operator

    It can be expressed as follows:

    .. math::

        q_{t}(x, t) = f\left(\left[P, E\right](x, t), m_{lt}(x, t), \left[c_i, c_p, c_t, c_l, k_{exc}\right](x), \left[h_i, h_p, h_t, h_l\right](x, t)\right)

    with :math:`q_{t}` the elemental discharge, :math:`P` the precipitation, :math:`E` the potential evapotranspiration,
    :math:`m_{lt}` the melt flux from the snow operator, :math:`c_i` the maximum capacity of the interception reservoir,
    :math:`c_p` the maximum capacity of the production reservoir, :math:`c_t` the maximum capacity of the transfer reservoir,
    :math:`c_l` the maximum capacity of the [slow-]transfer reservoir, :math:`k_{exc}` the exchange coefficient,
    :math:`h_i` the state of the interception reservoir, :math:`h_p` the state of the production reservoir,
    :math:`h_t` the state of the first transfer reservoir and :math:`h_l` the state of the second transfer reservoir.

    .. note::

        Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
        
        - Internal fluxes, :math:`\{q_{t}, m_{lt}\}\in\boldsymbol{q}`
        - Atmospheric forcings, :math:`\{P, E\}\in\boldsymbol{\mathcal{I}}`
        - Parameters, :math:`\{c_i, c_p, c_t, c_l, k_{exc}\}\in\boldsymbol{\theta}`
        - Normalized states: :math:`\{\tilde{h_i}, \tilde{h_p}, \tilde{h_t}, \tilde{h_l}\}`, where :math:`\tilde{h_i} = \frac{h_i}{c_i}`, :math:`\tilde{h_p} = \frac{h_p}{c_p}`, :math:`\tilde{h_t} = \frac{h_t}{c_t}`, and :math:`\tilde{h_l} = \frac{h_l}{c_l}`, with states :math:`\{h_i, h_p, h_t, h_l\} \in \boldsymbol{h}`

    The function :math:`f` is resolved numerically as follows:

    **Interception**

    Same as ``gr4`` interception, see :ref:`GR4 Interception <math_num_documentation.forward_structure.hydrological_module.gr4>`.

    **Production**

    Same as ``gr4`` production, see :ref:`GR4 Production <math_num_documentation.forward_structure.hydrological_module.gr4>`.

    **Exchange**

    Same as ``gr4`` exchange, see :ref:`GR4 Exchange <math_num_documentation.forward_structure.hydrological_module.gr4>`.

    **Transfer**

    - Split the production runoff :math:`p_r` into three branches (first transfer, second transfer and direct), :math:`p_{rr}`, :math:`p_{rl}` and :math:`p_{rd}`

    .. math::
        :nowrap:

        \begin{eqnarray}
            &p_{rr}(x, t)& &=& &0.6 \times 0.9(p_r(x, t) + p_{erc}(x, t)) + l_{exc}(x, t)\\
            &p_{rl}(x, t)& &=& &0.4 \times 0.9(p_r(x, t) + p_{erc}(x, t)) \\
            &p_{rd}(x, t)& &=& &0.1(p_r(x, t) + p_{erc}(x, t))
        \end{eqnarray}

    - Update the normalized transfer reservoir states :math:`\tilde{h_t}` and :math:`\tilde{h_l}`

    .. math::
        
        \tilde{h_t}(x, t^*) = \max\left(0, \tilde{h_t}(x, t - 1) + \frac{p_{rr}(x, t)}{c_t(x)}\right)

    .. math::

        \tilde{h_l}(x, t^*) = \max\left(0, \tilde{h_l}(x, t - 1) + \frac{p_{rl}(x, t)}{c_l(x)}\right)

    - Compute the transfer branch elemental discharges :math:`q_r` and :math:`q_l`

    .. math::
        :nowrap:

        \begin{eqnarray}
            &q_r(x, t)& &=& &\tilde{h_t}(x, t^*)c_t(x) - \left(\left(\tilde{h_t}(x, t^*)c_t(x)\right)^{-4} + c_t(x)^{-4}\right)^{-1/4}\\
            &q_l(x, t)& &=& &\tilde{h_l}(x, t^*)c_l(x) - \left(\left(\tilde{h_l}(x, t^*)c_l(x)\right)^{-4} + c_l(x)^{-4}\right)^{-1/4}
        \end{eqnarray}

    - Update the normalized transfer reservoir states :math:`\tilde{h_t}` and :math:`\tilde{h_l}`

    .. math::

        \tilde{h_t}(x, t) = \tilde{h_t}(x, t^*) - \frac{q_r(x, t)}{c_t(x)}
        
    .. math::
        
        \tilde{h_l}(x, t) = \tilde{h_l}(x, t^*) - \frac{q_l(x, t)}{c_l(x)}
        
    - Compute the direct branch elemental discharge :math:`q_d`

    .. math::

        q_d(x, t) = \max(0, p_{rd}(x, t) + l_{exc}(x, t))

    - Compute the elemental discharge :math:`q_t`

    .. math::

        q_t(x, t) = q_r(x, t) + q_l(x, t) + q_d(x, t)

.. _math_num_documentation.forward_structure.hydrological_module.grd:

.. dropdown:: Génie Rural Distribué (grd)
    :animate: fade-in-slide-down

    This hydrological operator is derived from the GR models and is a simplified structure used in :cite:t:`jay2019potential`.

    .. figure:: ../_static/grd_structure.svg
        :align: center
        :width: 300
        
        Diagram of the ``grd`` hydrological operator, a simplified ``GR`` like

    It can be expressed as follows:

    .. math::

        q_{t}(x, t) = f\left(\left[P, E\right](x, t), m_{lt}(x, t), \left[c_p, c_t\right](x), \left[h_p, h_t\right](x, t)\right)

    with :math:`q_{t}` the elemental discharge, :math:`P` the precipitation, :math:`E` the potential evapotranspiration,
    :math:`m_{lt}` the melt flux from the snow operator, :math:`c_p` the maximum capacity of the production reservoir, 
    :math:`c_t` the maximum capacity of the transfer reservoir, :math:`h_p` the state of the production reservoir and
    :math:`h_t` the state of the transfer reservoir.

    .. note::

        Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
        
        - Internal fluxes: :math:`\{q_{t}, m_{lt}\}\in\boldsymbol{q}`
        - Atmospheric forcings: :math:`\{P, E\}\in\boldsymbol{\mathcal{I}}`
        - Parameters: :math:`\{c_p, c_t\}\in\boldsymbol{\theta}`
        - Normalized states: :math:`\{\tilde{h_p}, \tilde{h_t}\}`, where :math:`\tilde{h_p} = \frac{h_p}{c_p}` and :math:`\tilde{h_t} = \frac{h_t}{c_t}`, with states :math:`\{h_p, h_t\} \in \boldsymbol{h}`

    The function :math:`f` is resolved numerically as follows:

    **Interception**

    - Compute the interception evapotranspiration :math:`e_i`

    .. math::

        e_i(x, t) = \min(E(x, t), P(x, t) + m_{lt}(x, t))

    - Compute the neutralized precipitation :math:`p_n` and evapotranspiration :math:`e_n`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &p_n(x, t)& &=& &\max \left(0, \; P(x, t) + m_{lt}(x, t) - e_i(x, t) \right)\\

            &e_n(x, t)& &=& &E(x, t) - e_i(x, t)

        \end{eqnarray}

    **Production**

    Same as ``gr4`` production, see :ref:`GR4 Production <math_num_documentation.forward_structure.hydrological_module.gr4>`.

    **Transfer**

    - Update the normalized transfer reservoir state :math:`\tilde{h_t}`

    .. math::
        
        \tilde{h_t}(x, t^*) = \max\left(0, \tilde{h_t}(x, t - 1) + \frac{p_{r}(x, t)}{c_t(x)}\right)

    - Compute the transfer branch elemental discharge :math:`q_r`

    .. math::
        :nowrap:

        \begin{eqnarray}

            q_r(x, t) = \tilde{h_t}(x, t^*)c_t(x) - \left(\left(\tilde{h_t}(x, t^*)c_t(x)\right)^{-4} + c_t(x)^{-4}\right)^{-1/4}

        \end{eqnarray}

    - Update the normalized transfer reservoir state :math:`\tilde{h_t}`

    .. math::

        \tilde{h_t}(x, t) = \tilde{h_t}(x, t^*) - \frac{q_r(x, t)}{c_t(x)}

    - Compute the elemental discharge :math:`q_t`

    .. math::

        q_t(x, t) = q_r(x, t)

.. _math_num_documentation.forward_structure.hydrological_module.loieau:

.. dropdown:: Génie Rural LoiEau (loieau)
    :animate: fade-in-slide-down

    This hydrological operator is derived from the GR model :cite:p:`Folton_2020`.

    .. hint::

        Helpful links about LoiEau:

        - `Database <https://loieau.recover.inrae.fr/>`__

    .. figure:: ../_static/loieau_structure.svg
        :align: center
        :width: 300
        
        Diagram of the ``loieau`` like hydrological operator

    It can be expressed as follows:

    .. math::

        q_{t}(x, t) = f\left(\left[P, E\right](x, t), m_{lt}(x, t), \left[c_a, c_c, k_b\right](x), \left[h_a, h_c\right](x, t)\right)

    with :math:`q_{t}` the elemental discharge, :math:`P` the precipitation, :math:`E` the potential evapotranspiration,
    :math:`m_{lt}` the melt flux from the snow operator, :math:`c_a` the maximum capacity of the production reservoir, 
    :math:`c_c` the maximum capacity of the transfer reservoir, :math:`k_b` the transfer coefficient, 
    :math:`h_a` the state of the production reservoir and :math:`h_c` the state of the transfer reservoir.

    .. note::

        Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
        
        - Internal fluxes: :math:`\{q_{t}, m_{lt}\}\in\boldsymbol{q}`
        - Atmospheric forcings: :math:`\{P, E\}\in\boldsymbol{\mathcal{I}}`
        - Parameters: :math:`\{c_a, c_c, k_b\}\in\boldsymbol{\theta}`
        - Normalized states: :math:`\{\tilde{h_a}, \tilde{h_c}\}`, where :math:`\tilde{h_a} = \frac{h_a}{c_a}` and :math:`\tilde{h_c} = \frac{h_c}{c_c}`, with states :math:`\{h_a, h_c\} \in \boldsymbol{h}`

    The function :math:`f` is resolved numerically as follows:

    **Interception**

    Same as ``grd`` interception, see :ref:`GRD Interception <math_num_documentation.forward_structure.hydrological_module.grd>`.

    **Production**

    Same as ``gr4`` production, see :ref:`GR4 Production <math_num_documentation.forward_structure.hydrological_module.gr4>`.

    .. note::

        The parameter :math:`c_p` is replaced by :math:`c_a` and the state :math:`h_p` by :math:`h_a`

    **Transfer**

    - Split the production runoff :math:`p_r` into two branches (transfer and direct), :math:`p_{rr}` and :math:`p_{rd}`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &p_{rr}(x, t)& &=& &0.9(p_r(x, t) + p_{erc}(x, t))\\
            &p_{rd}(x, t)& &=& &0.1(p_r(x, t) + p_{erc}(x, t))

        \end{eqnarray}

    - Update the normalized transfer reservoir state :math:`\tilde{h_c}`

    .. math::
        
        \tilde{h_c}(x, t^*) = \max\left(0, \tilde{h_c}(x, t - 1) + \frac{p_{rr}(x, t)}{c_c(x)}\right)

    - Compute the transfer branch elemental discharge :math:`q_r`

    .. math::
        :nowrap:

        \begin{eqnarray}

            q_r(x, t) = \tilde{h_c}(x, t^*)c_c(x) - \left(\left(\tilde{h_c}(x, t^*)c_c(x)\right)^{-3} + c_c(x)^{-3}\right)^{-1/3}

        \end{eqnarray}

    - Update the normalized transfer reservoir state :math:`\tilde{h_c}`

    .. math::

        \tilde{h_c}(x, t) = \tilde{h_c}(x, t^*) - \frac{q_r(x, t)}{c_c(x)}

    - Compute the direct branch elemental discharge :math:`q_d`

    .. math::

        q_d(x, t) = \max(0, p_{rd}(x, t))

    - Compute the elemental discharge :math:`q_t`

    .. math::

        q_t(x, t) = k_b(x)\left(q_r(x, t) + q_d(x, t)\right)

.. _math_num_documentation.forward_structure.hydrological_module.gr_rainfall_intensity:

.. dropdown:: Génie Rural with rainfall intensity terms (gr4_ri, gr5_ri)

    .. _math_num_documentation.forward_structure.hydrological_module.gr4_ri:

    .. dropdown:: gr4_ri
        :animate: fade-in-slide-down

        This hydrological module is derived from the model introduced in :cite:t:`Astagneau_2022`.

        .. figure:: ../_static/gr4-ri_structure.svg
            :align: center
            :width: 400
            
            Diagram of the ``gr4_ri`` like hydrological operator

        It can be expressed as follows:

        .. math::

            q_{t}(x, t) = f\left(\left[P, E\right](x, t), m_{lt}(x, t), \left[c_i, c_p, c_t, \alpha_1, \alpha_2, k_{exc}\right](x), \left[h_i, h_p, h_t\right](x, t)\right)

        with :math:`q_{t}` the elemental discharge, :math:`P` the precipitation, :math:`E` the potential evapotranspiration,
        :math:`m_{lt}` the melt flux from the snow operator, :math:`c_i` the maximum capacity of the interception reservoir,
        :math:`c_p` the maximum capacity of the production reservoir, :math:`c_t` the maximum capacity of the transfer reservoir,
        :math:`k_{exc}` the exchange coefficient, :math:`h_i` the state of the interception reservoir, 
        :math:`h_p` the state of the production reservoir and :math:`h_t` the state of the transfer reservoir,
        :math:`\alpha_1` and :math:`\alpha_2` parameters controling the rainfall intensity rate respectively in time unit per :math:`mm` and in :math:`mm` per time unit.

        .. note::

            Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
            
            - Internal fluxes: :math:`\{q_{t}, m_{lt}\}\in\boldsymbol{q}`
            - Atmospheric forcings: :math:`\{P, E\}\in\boldsymbol{\mathcal{I}}`
            - Parameters: :math:`\{c_i, c_p, c_t, \alpha_1, \alpha_2, k_{exc}\}\in\boldsymbol{\theta}`
            - States: :math:`\{h_i, h_p, h_t\} \in \boldsymbol{h}`
        
        The function :math:`f` is resolved numerically as follows:

        **Interception**

        Same as ``gr4`` interception, see :ref:`GR4 Interception <math_num_documentation.forward_structure.hydrological_module.gr4>`.

        **Production** 

        In the classical gr production reservoir formulation, the instantaneous production rate is the ratio between the state and the capacity of the reservoir,
        :math:`\eta = \left( \frac{h_p}{c_p} \right)^2`. 
        The infiltration flux :math:`p_s` is obtained by temporal integration as follows:

        .. math::
            :nowrap:

            \begin{eqnarray}

                &p_s = \int_{t-\Delta t}^{t} (1 - \eta) dt \\
            
            \end{eqnarray}
            
        Assuming the neutralized rainfall :math:`p_n` constant over the current time step and thanks to analytically integrable function, the infiltration flux into the production reservoir is obtained:

        .. math::
            :nowrap:
            
            \begin{eqnarray}

                &p_s = & c_p \tanh\left(\frac{p_n}{c_p}\right) \frac{1 - \left( \frac{h_p}{c_p} \right)^2}{1 + \frac{h_p}{c_p} \tanh\left( \frac{p_n}{c_p} \right)} \\
                
            \end{eqnarray}

        To improve runoff production by a gr reservoir, 
        even with low production level in dry condition, 
        in the case of high rainfall intensity, in :cite:t:`Astagneau_2022` they suggest a modification 
        of the infiltration rate :math:`p_s` depending on rainfall intensity :math:`p_n`. 
        Indeed, let's consider the rainfall intensity coefficient :math:`\gamma`,
        function of weighted rainfall intensity.

        .. math::
            :nowrap:

            \begin{eqnarray}

                & \gamma = & 1 - \exp(-p_n \times \alpha_1) \\
            
            \end{eqnarray}
        
        with :math:`\alpha_1` in time unit per :math:`mm`.

        The expression of the instantaneous production rate changes as follows

        .. math::
            :nowrap:

            \begin{eqnarray}

                & \eta = & \left( 1 - \gamma \right) \left( \frac{h_p}{c_p} \right)^2 + \gamma \\
            
            \end{eqnarray}

        Thus the infiltration rate becomes

        .. math::
            :nowrap:

            \begin{eqnarray}

            &p_s& &=& &\int_{t-\Delta t}^{t} (1 - \eta) dt\\

            && &=& &\int_{t-\Delta t}^{t} \left(1 - (1-\gamma) \left(\frac{h_p}{c_p} \right)^2 \right) dt - \int_{t-\Delta t}^{t} \gamma dt\\
            
            && &=& &\left[ \frac{ c_p }{ \sqrt{1-\gamma} } \tanh \left( \frac{\sqrt{1-\gamma} \  h_p}{c_p} \right) \right]_{t-\Delta t}^t - \gamma \Delta t
            
            \end{eqnarray}


        We denote :math:`\lambda = \sqrt{1 - \gamma}`, then

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

            Note that if :math:`\alpha_1 = 0`, we return to the general writing of the instantaneous production rate.

        **Exchange**

        Same as ``gr4`` exchange, see :ref:`GR4 Exchange <math_num_documentation.forward_structure.hydrological_module.gr4>`.  
            
        **Transfer**
        
        In context of high rainfall intensities triggering flash flood responses, it is crucial to account for fast dynamics related to surface/hypodermic runoff 
        and slower responses due to delayed/deeper flows (e.g. :cite:t:`Douinot_2018_multihypothesis`). 
        Following :cite:t:`Astagneau_2022` for a lumped GR model, we introduce at pixel scale in `smash` a function to modify the partitioning between fast 
        and slower transfert branches depending on rainfall intensity of the current time step only (small pixel size):
        
        .. math::
            :nowrap:

            \begin{eqnarray}

                &p_{rr}& =& (1 - spl)(p_r + p_{erc}) + l_{exc}\\
                &p_{rd}& =& spl(p_r + p_{erc}) \\
                &spl& =& 0.9 \tanh(\alpha_2 p_n)^2 + 0.1
                
            \end{eqnarray}

        with :math:`\alpha_2` in :math:`mm` per time unit.

        .. note::

            If :math:`\alpha_2 = 0`, we return to the ``gr-4`` writing of the transfer.
            If :math:`\alpha_2 = \alpha_1 = 0`, it is equivalent to ``gr-4`` structure.


    .. _math_num_documentation.forward_structure.hydrological_module.gr5_ri:

    .. dropdown:: gr5_ri
        :animate: fade-in-slide-down

        This hydrological module is derived from the model introduced in :cite:t:`Astagneau_2022`.

        .. figure:: ../_static/gr5-ri_structure.svg
            :align: center
            :width: 400
            
            Diagram of the ``gr5_ri`` like hydrological operator

        It can be expressed as follows:

        .. math::

            q_{t}(x, t) = f\left(\left[P, E\right](x, t), m_{lt}(x, t), \left[c_i, c_p, c_t, \alpha_1, \alpha_2, k_{exc}, a_{exc}\right](x), \left[h_i, h_p, h_t\right](x, t)\right)

        with :math:`q_{t}` the elemental discharge, :math:`P` the precipitation, :math:`E` the potential evapotranspiration,
        :math:`m_{lt}` the melt flux from the snow operator, :math:`c_i` the maximum capacity of the interception reservoir,
        :math:`c_p` the maximum capacity of the production reservoir, :math:`c_t` the maximum capacity of the transfer reservoir,
        :math:`k_{exc}` the exchange coefficient, :math:`a_{exc}` the exchange threshold, :math:`h_i` the state of the interception reservoir, 
        :math:`h_p` the state of the production reservoir and :math:`h_t` the state of the transfer reservoir,
        :math:`\alpha_1` and :math:`\alpha_2` parameters controling the rainfall intensity rate respectively in time unit per :math:`mm` and in :math:`mm` per time unit.

        .. note::

            Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
            
            - Internal fluxes: :math:`\{q_{t}, m_{lt}\}\in\boldsymbol{q}`
            - Atmospheric forcings: :math:`\{P, E\}\in\boldsymbol{\mathcal{I}}`
            - Parameters: :math:`\{c_i, c_p, c_t, \alpha_1, \alpha_2, k_{exc}, a_{exc}\}\in\boldsymbol{\theta}`
            - States: :math:`\{h_i, h_p, h_t\} \in \boldsymbol{h}`
        
        The function :math:`f` is resolved numerically as follows:

        **Interception**

        Same as ``gr4`` interception, see :ref:`GR4 Interception <math_num_documentation.forward_structure.hydrological_module.gr4>`.

        **Production** 

        Same as ``gr4_ri`` production, see :ref:`GR4 Production <math_num_documentation.forward_structure.hydrological_module.gr4>`.

        **Exchange**

        Same as ``gr5`` exchange, see :ref:`GR5 Exchange <math_num_documentation.forward_structure.hydrological_module.gr5>`. 
            
        **Transfer**
        
        Same as ``gr4_ri`` transfer, see :ref:`GR4 Transfer <math_num_documentation.forward_structure.hydrological_module.gr4>`.

.. _math_num_documentation.forward_structure.hydrological_module.hybrid_flux_correction:

.. dropdown:: Hybrid GR for flux correction (gr4_mlp, gr5_mlp, gr6_mlp, grc_mlp, grd_mlp, loieau_mlp)
    :animate: fade-in-slide-down

    These hydrological models are GR-like models embedded within a multilayer perceptron (MLP) to correct internal water fluxes. Such a neural network is referred to as a process-parameterization neural network. 
    This process-parameterization neural network takes as inputs the neutralized precipitation, the neutralized potential evapotranspiration, and the model states from the previous time step, and produces the corrections of internal water fluxes as outputs:

    .. math::

        \boldsymbol{f}_{q}(x,t) = \phi\left(\boldsymbol{I}_n(x,t),\boldsymbol{h}(x,t-1);\boldsymbol{\rho}\right)

    where :math:`\boldsymbol{f}_{q}` is the vector of flux corrections, :math:`\boldsymbol{I}_n` is the neutralized atmospheric forcings, :math:`\boldsymbol{h}` is the vector of model states, and :math:`\boldsymbol{\rho}` is the parameters of the process-parameterization neural network :math:`\phi`.

    .. note::
        The output layer of this neural network uses a ``TanH`` activation function to map the hydrological flux corrections to the range :math:`]-1, 1[`.

    .. dropdown:: gr4_mlp
        :animate: fade-in-slide-down

        This hydrological module is principally based on the ``gr4`` operators, with the integration of a neural network for correcting internal water fluxes as follows:

        **Interception**

        Same as ``gr4`` interception, see :ref:`GR4 Interception <math_num_documentation.forward_structure.hydrological_module.gr4>`.

        **Process-parameterization neural network**

        .. math::

            [f_p, f_e, f_c, f_l](x,t) = \phi\left([p_n, e_n](x,t), [\tilde{h_p}, \tilde{h_t}](x,t-1);\boldsymbol{\rho}\right)

        where :math:`p_n, e_n` are the neutralized precipitation and potential evapotranspiration obtained from interception; 
        :math:`\tilde{h_p}, \tilde{h_t}` are the normalized states of the production and transfer reservoirs; 
        :math:`f_p, f_e, f_c, f_l` are the corrections applied to internal fluxes as follows.

        **Production**

        Similar to ``gr4`` production, but the equations for computing infiltrating precipitation :math:`p_s` and evapotranspiration :math:`e_s` are updated to integrate flux corrections
    
        .. math::
            :nowrap:

            \begin{eqnarray}

            &p_s(x, t)& & := & &\min\left(p_n(x, t), (1 + f_p) p_s(x, t)\right) \\

            &e_s(x, t)& & := & &\min\left(e_n(x, t), (1 + f_e) e_s(x, t)\right)
            \end{eqnarray}

        **Exchange**

        Compute the refined exchange flux
    
        .. math::

            l_{exc}(x, t) = (1 + f_l) k_{exc}(x) \tilde{h_t}(x, t - 1)^{7/2}

        **Transfer**

        Same as ``gr4`` transfer except the equations of splitting the production runoff

        .. math::
            :nowrap:

            \begin{eqnarray}

                &p_{rr}(x, t)& &=& &(0.9 - 0.9 f_c^2)(p_r(x, t) + p_{erc}(x, t)) + l_{exc}(x, t)\\
                &p_{rd}(x, t)& &=& &(0.1 + 0.9 f_c^2)(p_r(x, t) + p_{erc}(x, t))

            \end{eqnarray}

    .. dropdown:: gr5_mlp
        :animate: fade-in-slide-down

        This hydrological module is principally based on the ``gr5`` operators, with the integration of a neural network for correcting internal water fluxes as follows:

        **Interception**

        Same as ``gr4`` interception.

        **Process-parameterization neural network**

        Same as ``gr4_mlp`` process-parameterization neural network.

        **Production**

        Same as ``gr4_mlp`` production.

        **Exchange**

        Compute the refined exchange flux

        .. math::

            l_{exc}(x, t) = (1 + f_l) k_{exc}(x) \left(\tilde{h_t}(x, t - 1) - a_{exc}(x)\right)

        **Transfer**

        Same as ``gr4_mlp`` transfer.

    .. dropdown:: gr6_mlp
        :animate: fade-in-slide-down

        This hydrological module is principally based on the ``gr6`` operators, with the integration of a neural network for correcting internal water fluxes as follows:

        **Interception**

        Same as ``gr4`` interception.

        **Process-parameterization neural network**

        .. math::

            [f_p, f_e, f_{c1}, f_{c2}, f_l](x,t) = \phi\left([p_n, e_n](x,t), [\tilde{h_p}, \tilde{h_t}, \tilde{h_e}](x,t-1);\boldsymbol{\rho}\right)

        where :math:`p_n, e_n` are the neutralized precipitation and potential evapotranspiration obtained from interception; 
        :math:`\tilde{h_p}, \tilde{h_t}, \tilde{h_e}` are the normalized states of the production, transfer, and exponential reservoirs; 
        :math:`f_p, f_e, f_{c1}, f_{c2}, f_l` are the corrections applied to internal fluxes as follows.

        **Production**

        Same as ``gr4_mlp`` production.

        **Exchange**

        Same as ``gr5_mlp`` exchange.

        **Transfer**

        Same as ``gr6`` transfer except the equations of splitting the production runoff

        .. math::
            :nowrap:

            \begin{eqnarray}
                &p_{rr}(x, t)& &=& &(0.6 - 0.4 f_{c2}) (0.9 - 0.9 f_{c1}^2) (p_r(x, t) + p_{erc}(x, t)) + l_{exc}(x, t)\\
                &p_{re}(x, t)& &=& &(0.4 + 0.4 f_{c2}) (0.9 - 0.9 f_{c1}^2) (p_r(x, t) + p_{erc}(x, t)) + l_{exc}(x, t)\\
                &p_{rd}(x, t)& &=& &(0.1 + 0.9 f_{c1}^2) (p_r(x, t) + p_{erc}(x, t))
            \end{eqnarray}

    .. dropdown:: grc_mlp
        :animate: fade-in-slide-down

        This hydrological module is principally based on the ``grc`` operators, with the integration of a neural network for correcting internal water fluxes as follows:

        **Interception**

        Same as ``gr4`` interception.

        **Process-parameterization neural network**

        .. math::

            [f_p, f_e, f_{c1}, f_{c2}, f_l](x,t) = \phi\left([p_n, e_n](x,t), [\tilde{h_p}, \tilde{h_t}, \tilde{h_l}](x,t-1);\boldsymbol{\rho}\right)

        where :math:`p_n, e_n` are the neutralized precipitation and potential evapotranspiration obtained from interception; 
        :math:`\tilde{h_p}, \tilde{h_t}, \tilde{h_l}` are the normalized states of the production, first transfer, and second transfer reservoirs; 
        :math:`f_p, f_e, f_{c1}, f_{c2}, f_l` are the corrections applied to internal fluxes as follows.

        **Production**

        Same as ``gr4_mlp`` production.

        **Exchange**

        Same as ``gr4_mlp`` exchange.

        **Transfer**

        Same as ``grc`` transfer except the equations of splitting the production runoff

        .. math::
            :nowrap:

            \begin{eqnarray}
                &p_{rr}(x, t)& &=& &(0.6 - 0.4 f_{c2}) (0.9 - 0.9 f_{c1}^2) (p_r(x, t) + p_{erc}(x, t)) + l_{exc}(x, t)\\
                &p_{rl}(x, t)& &=& &(0.4 + 0.4 f_{c2}) (0.9 - 0.9 f_{c1}^2) (p_r(x, t) + p_{erc}(x, t))\\
                &p_{rd}(x, t)& &=& &(0.1 + 0.9 f_{c1}^2) (p_r(x, t) + p_{erc}(x, t))
            \end{eqnarray}

    .. dropdown:: grd_mlp
        :animate: fade-in-slide-down

        This hydrological module is principally based on the ``grd`` operators, with the integration of a neural network for correcting internal water fluxes as follows:

        **Interception**

        Same as ``grd`` interception.

        **Process-parameterization neural network**

        .. math::

            [f_p, f_e](x,t) = \phi\left([p_n, e_n](x,t), [\tilde{h_p}, \tilde{h_t}](x,t-1);\boldsymbol{\rho}\right)

        where :math:`p_n, e_n` are the neutralized precipitation and potential evapotranspiration obtained from interception; 
        :math:`\tilde{h_p}, \tilde{h_t}` are the normalized states of the production and transfer reservoirs; 
        :math:`f_p, f_e` are the corrections applied to internal fluxes as follows.

        **Production**

        Same as ``gr4_mlp`` production.

        **Transfer**

        Same as ``grd`` transfer.

    .. dropdown:: loieau_mlp
        :animate: fade-in-slide-down

        This hydrological module is principally based on the ``loieau`` operators, with the integration of a neural network for correcting internal water fluxes as follows:

        **Interception**

        Same as ``grd`` interception.

        **Process-parameterization neural network**

        .. math::

            [f_p, f_e, f_c](x,t) = \phi\left([p_n, e_n](x,t), [\tilde{h_a}, \tilde{h_c}](x,t-1);\boldsymbol{\rho}\right)

        where :math:`p_n, e_n` are the neutralized precipitation and potential evapotranspiration obtained from interception; 
        :math:`\tilde{h_a}, \tilde{h_c}` are the normalized states of the production and transfer reservoirs; 
        :math:`f_p, f_e, f_c` are the corrections applied to internal fluxes as follows.

        **Production**

        Same as ``gr4_mlp`` production.

        .. note::
            The parameter :math:`c_p` is replaced by :math:`c_a` and the state :math:`h_p` by :math:`h_a`

        **Transfer**

        Same as ``loieau`` transfer except the equations of splitting the production runoff

        .. math::
            :nowrap:

            \begin{eqnarray}

                &p_{rr}(x, t)& &=& &(0.9 - 0.9 f_c^2)(p_r(x, t) + p_{erc}(x, t))\\
                &p_{rd}(x, t)& &=& &(0.1 + 0.9 f_c^2)(p_r(x, t) + p_{erc}(x, t))

            \end{eqnarray}

.. _math_num_documentation.forward_structure.hydrological_module.continuous_state_space_gr:

.. dropdown:: Continuous state-space Génie Rural with 4 parameters (gr4_ode)
    :animate: fade-in-slide-down

    This continuous state-space representation of the GR4 model is adapted from :cite:t:`santos2018continuous` for both lumped and distributed models. 
    Instead of relying on an algebraic approach based on an analytical solution, this representation involves numerically solving the ordinary differential equations (ODEs) of the GR4 model:

    .. math::
        \frac{d\boldsymbol{h}}{dt}=\left(\begin{array}{c} 
        \frac{dh_{p}}{dt}\\
        \frac{dh_{t}}{dt}
        \end{array}\right)
        = \left(
        \begin{array}{c}
        \left(1-\tilde{h_p}^{\alpha_1}\right)p_n-\tilde{h_p}(2-\tilde{h_p})e_n \\
        0.9\tilde{h_p}^{\alpha_1}p_n+k_{exc}\tilde{h_t}^{\alpha_3}- \frac{c_t}{\alpha_2-1}\tilde{h_t}^{\alpha_2}
        \end{array}
        \right)

    where :math:`\alpha_1=2, \alpha_2=5, \alpha_3=3.5` are classical GR constants (cf. :cite:t:`perrin2003improvement, santos2018continuous`); 
    :math:`p_n, e_n` are the neutralized precipitation and potential evapotranspiration obtained from interception; 
    :math:`\tilde{h_p}, \tilde{h_t}` are the normalized states of the production and transfer reservoirs.

    This ODE system is solved using an implicit Euler scheme, where the Newton-Raphson method is used to approximate the sought states with a Jacobian matrix explicitly computed.
    
    Then, hydrological runoff flux (lateral discharge feeding the routing module) produced at the pixel scale is computed by the closure equation of the ODE system as follows:

    .. math::
        q_{t} = 0.1\tilde{h_p}^{\alpha_1}p_n + k_{exc}\tilde{h_t}^{\alpha_3} + \frac{c_t}{\alpha_2-1}\tilde{h_t}^{\alpha_2}

.. _math_num_documentation.forward_structure.hydrological_module.hybrid_neural_ode:

.. dropdown:: Hybrid GR4 with neural ODEs (gr4_ode_mlp)
    :animate: fade-in-slide-down

    This hybrid continuous state-space model is embedded within an MLP into the ODEs for process-parameterization.
    This process-parameterization neural network takes as inputs the neutralized precipitation, the neutralized potential evapotranspiration, and the model states from the previous time step, and produces the corrections of internal water fluxes as outputs:
    
    .. math::

        [f_p, f_e, f_t, f_l](x,t) = \phi\left([p_n, e_n](x,t), [\tilde{h_p}, \tilde{h_t}](x,t-1);\boldsymbol{\rho}\right)

    where :math:`p_n, e_n` are the neutralized precipitation and potential evapotranspiration obtained from interception; 
    :math:`\tilde{h_p}, \tilde{h_t}` are the normalized states of the production and transfer reservoirs; 
    :math:`f_p, f_e, f_t, f_l` are the corrections applied to the ODE system as follows.
    
    .. math::
        \frac{d\boldsymbol{h}}{dt}=\left(\begin{array}{c} 
        \frac{dh_{p}}{dt}\\
        \frac{dh_{t}}{dt}
        \end{array}\right)
        = \left(
        \begin{array}{c}
        \left(1-\tilde{h_p}^{\alpha_1}\right)p_n(1+f_p)-\tilde{h_p}(2-\tilde{h_p})e_n(1+f_e) \\
        0.9\tilde{h_p}^{\alpha_1}p_n(1+f_p)+k_{exc}\tilde{h_t}^{\alpha_3}(1+f_l)- \frac{c_t}{\alpha_2-1}\tilde{h_t}^{\alpha_2}(1+f_t)
        \end{array}
        \right)

    where :math:`\alpha_1=2, \alpha_2=5, \alpha_3=3.5` are classical GR constants (cf. :cite:t:`perrin2003improvement, santos2018continuous`).

    .. note::
        The output layer of this neural network uses a ``TanH`` activation function to map the hydrological flux corrections to the range :math:`]-1, 1[`. 
        The hidden layer(s) use a ``SiLU`` function, which is twice differentiable everywhere and provides smooth gradients. 
        This is essential because the process-parameterization network must be twice differentiable—once for solving the ODEs and once for the calibration process to ensure numerical consistency during optimization—particularly as we aim to preserve the original structure by producing outputs close to zero at the beginning of optimization.


    This ODE system is solved using an implicit Euler scheme, where the Newton-Raphson method is used to approximate the sought states with a Jacobian matrix explicitly computed. 
    The Jacobian matrix is computed using the chain rule, where the derivatives of the neural network are computed using the backpropagation algorithm.
    
    Then, hydrological runoff flux (lateral discharge feeding the routing module) produced at the pixel scale is computed by the closure equation of the ODE system as follows:

    .. math::
        q_{t} = 0.1\tilde{h_p}^{\alpha_1}p_n(1+f_p) + k_{exc}\tilde{h_t}^{\alpha_3}(1+f_l) + \frac{c_t}{\alpha_2-1}\tilde{h_t}^{\alpha_2}(1+f_t)

.. _math_num_documentation.forward_structure.hydrological_module.imperviousness:

.. dropdown:: Génie Rural with imperviousness 
    :animate: fade-in-slide-down

    This imperviousness feature allows for the calculation of the impervious proportion of a pixel's surface and takes this into account when computing infiltration and evapotranspiration fluxes applied to the GR type production reservoir.
    The imperviousness coefficients  :math:`imperv(x)` influence the fluxes of the production reservoir of each cell by being applied to the neutralized rainfall :math:`p_n(x,t)` and the evapotranspiration :math:`e_s(x,t)`.
    The imperviousness coefficients must range between 0 and 1 and be specified through an input map that is consistent with the model grid. This map can be obtained, for example, from soil occupation processing.
    For instance, if the imperviousness coefficient is close to 1, the production part receives less neutralized rainfall :math:`p_n` and there is less evapotranspiration :math:`e_s` from the impermeable soil.
    This imperviousness accounting for the GR reservoir is applicable to GR model structures in `smash`. This is illustrated here on the GR4 structure.

    .. figure:: ../_static/gr4_structure_imperviousness.svg
        :align: center
        :width: 300
        
        Diagram of the ``gr4`` hydrological operator with imperviousness, a simplified ``GR`` like model for spatialized modeling.

    **Production**

    - Compute the neutralized precipitation :math:`p_n` on impermeable soil

    .. math::
        :nowrap:

        \begin{eqnarray}

            &p_n(x, t)& &=& & \left(1 - imperv(x)\right)\ p_n(x, t)

        \end{eqnarray}
    
    - Compute the production infiltrating precipitation :math:`p_s` and evapotranspiration :math:`e_s`

    .. math::
        :nowrap:

        \begin{eqnarray}

        &p_s(x, t)& &=& &c_p(x) (1 - h_p(x, t - 1)^2) \frac{\tanh\left(\frac{p_n(x, t)}{c_p(x)}\right)}{1 + h_p(x, t - 1) \tanh\left(\frac{p_n(x, t)}{c_p(x)}\right)}\\

        &e_s(x, t)& &=& &(1 - imperv(x)) \left(h_p(x, t - 1) c_p(x) (2 - h_p(x, t - 1)) \frac{\tanh\left(\frac{e_n(x, t)}{c_p(x)}\right)}{1 + (1 - h_p(x, t - 1)) \tanh\left(\frac{e_n(x, t)}{c_p(x)}\right)} \right)
        \end{eqnarray}

.. _math_num_documentation.forward_structure.hydrological_module.vic3l:

.. dropdown:: Variable Infiltration Curve 3 Layers (vic3l)
    :animate: fade-in-slide-down

    This hydrological operator is derived from the VIC model :cite:p:`liang1994simple`.

    .. hint::

        Helpful links about VIC:

        - `Model overview <https://vic.readthedocs.io/en/master/Overview/ModelOverview/>`__
        - `References <https://vic.readthedocs.io/en/master/Documentation/References/>`__
        - `GitHub <https://github.com/UW-Hydro/VIC/>`__

    .. figure:: ../_static/vic3l_structure.svg
        :align: center
        :width: 300
        
        Diagram of the ``vic3l`` like hydrological operator

    It can be expressed as follows:

    .. math::

        q_{t}(x, t) = f\left(\left[P, E\right](x, t), m_{lt}(x, t), \left[b, c_{usl}, c_{msl}, c_{bsl}, k_s, p_{bc}, d_{sm}, d_s, w_s\right](x), \left[h_{cl}, h_{usl}, h_{msl}, h_{bsl}\right](x, t)\right)

    with :math:`q_{t}` the elemental discharge, :math:`P` the precipitation, :math:`E` the potential evapotranspiration,
    :math:`m_{lt}` the melt flux from the snow operator, :math:`b` the variable infiltration curve parameter,
    :math:`c_{usl}` the maximum capacity of the upper soil layer, :math:`c_{msl}` the maximum capacity of the medium soil layer,
    :math:`c_{bsl}` the maximum capacity of the bottom soil layer, :math:`k_s` the saturated hydraulic conductivity,
    :math:`p_{bc}` the Brooks and Corey exponent, :math:`d_{sm}` the maximum velocity of baseflow, 
    :math:`d_s` the non-linear baseflow threshold maximum velocity, :math:`w_s` the non-linear baseflow threshold soil moisture,
    :math:`h_{cl}` the state of the canopy layer, :math:`h_{usl}` the state of the upper soil layer,
    :math:`h_{msl}` the state of the medium soil layer and :math:`h_{bsl}` the state of the bottom soil layer. 

    .. note::

        Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
        
        - Internal fluxes: :math:`\{q_{t}, m_{lt}\}\in\boldsymbol{q}`
        - Atmospheric forcings: :math:`\{P, E\}\in\boldsymbol{\mathcal{I}}`
        - Parameters: :math:`\{b, c_{usl}, c_{msl}, c_{bsl}, k_s, p_{bc}, d_{sm}, d_s, w_s\}\in\boldsymbol{\theta}`
        - Normalized states: :math:`\{\tilde{h_{cl}}, \tilde{h_{usl}}, \tilde{h_{msl}}, \tilde{h_{bsl}}\}`, where :math:`\tilde{h_{cl}} = \frac{h_{cl}}{c_{usl}}`, :math:`\tilde{h_{usl}} = \frac{h_{usl}}{c_{usl}}`, :math:`\tilde{h_{msl}} = \frac{h_{msl}}{c_{msl}}`, and :math:`\tilde{h_{bsl}} = \frac{h_{bsl}}{c_{bsl}}`, with states :math:`\{h_{cl}, h_{usl}, h_{msl}, h_{bsl}\} \in \boldsymbol{h}`

    The function :math:`f` is resolved numerically as follows:

    **Canopy layer interception**

    - Compute the canopy layer interception evapotranspiration :math:`e_c`

    .. math::

        e_c(x, t) = \min(E(x, t)\tilde{h_{cl}}(x, t - 1)^{2/3}, P(x, t) + m_{lt}(x, t) + \tilde{h_{cl}}(x, t - 1))

    - Compute the neutralized precipitation :math:`p_n` and evapotranspiration :math:`e_n`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &p_n(x, t)& &=& &\max\left(0, P(x, t) + m_{lt}(x, t) - (1 - \tilde{h_{cl}}(x, t - 1)) - e_c(x, t)\right)\\
            &e_n(x, t)& &=& &E(x, t) - e_c(x, t)

        \end{eqnarray}

    - Update the normalized canopy layer interception state :math:`\tilde{h_{cl}}`

    .. math::

        \tilde{h_{cl}}(x, t) = \tilde{h_{cl}}(x, t - 1) + P(x, t) - e_c(x, t) - p_n(x, t)

    **Upper soil layer evapotranspiration**

    - Compute the maximum :math:`i_{m}` and the corresponding soil saturation :math:`i_{0}` infiltration

    .. math::
        :nowrap:

        \begin{eqnarray}

            &i_{m}(x, t)& &=& &(1 + b(x))c_{usl}(x)\\
            &i_{0}(x, t)& &=& &i_{m}(x, t)\left(1 - (1 - \tilde{h_{usl}}(x, t - 1))^{1/(1 - b(x))}\right)

        \end{eqnarray}

    - Compute the upper soil layer evapotranspiration :math:`e_s`

    .. math::
        :nowrap:

        \begin{eqnarray}

            e_s(x, t) =
            \begin{cases}

                e_n(x, t) &\text{if} \; i_{0}(x, t) \geq i_{m}(x, t) \\
                \beta(x, t)e_n(x, t) &\text{otherwise}

            \end{cases}

        \end{eqnarray}

    with :math:`\beta`, the beta function in the ARNO evapotranspiration :cite:p:`todini1996arno` (Appendix A)

    .. FIXME Maybe explain what is the beta function, power expansion ...

    - Update the normalized upper soil layer reservoir state :math:`\tilde{h_{usl}}`

    .. math::

        \tilde{h_{usl}}(x, t) = \tilde{h_{usl}}(x, t - 1) - \frac{e_s(x, t)}{c_{usl}(x)}

    **Infiltration**

    - Compute the maximum capacity :math:`c_{umsl}`, the soil moisture :math:`w_{umsl}` and the relative state :math:`h_{umsl}` of the first two layers

    .. math::
        :nowrap:

        \begin{eqnarray}

            &c_{umsl}(x)& &=& &c_{usl}(x) + c_{msl}(x)\\
            &w_{umsl}(x, t - 1)& &=& &\tilde{h_{usl}}(x, t - 1)c_{usl}(x) + \tilde{h_{msl}}(x, t - 1)c_{msl}(x)\\
            &h_{umsl}(x, t - 1)& &=& &\frac{w_{umsl}(x, t - 1)}{c_{umsl}(x)}

        \end{eqnarray}

    - Compute the maximum :math:`i_{m}` and the corresponding soil saturation :math:`i_{0}` infiltration

    .. math::
        :nowrap:

        \begin{eqnarray}

            &i_{m}(x, t)& &=& &(1 + b(x))c_{umsl}(x)\\
            &i_{0}(x, t)& &=& &i_{m}(x, t)\left(1 - (1 - h_{umsl}(x, t - 1))^{1/(1 - b(x))}\right)

        \end{eqnarray}

    - Compute the infiltration :math:`i`

    .. math::
        :nowrap:

        \begin{eqnarray}

            i(x, t) = 
            \begin{cases}

                c_{umsl}(x) - w_{umsl}(x, t - 1) &\text{if} \; i_{0}(x, t) + p_n(x, t) > i_{m}(x, t) \\
                c_{umsl}(x) - w_{umsl}(x, t - 1) - c_{umsl}(x)\left(1 - \frac{i_{0}(x, t) + p_n(x, t)}{i_m(x, t)}\right)^{b(x) + 1} &\text{otherwise}

            \end{cases}

        \end{eqnarray}

    - Distribute the infiltration :math:`i` between the first two layers, :math:`i_{usl}` and :math:`i_{msl}`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &i_{usl}(x, t)& &=& &\min((1 - \tilde{h_{usl}}(x, t - 1)c_{usl}(x), i(x, t))\\
            &i_{msl}(x, t)& &=& &\min((1 - \tilde{h_{msl}}(x, t - 1)c_{msl}(x), i(x, t) - i_{usl}(x, t))

        \end{eqnarray}

    - Update the first two layers reservoir states normalized, :math:`\tilde{h_{usl}}` and :math:`\tilde{h_{msl}}`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &\tilde{h_{usl}}(x, t)& &=& &\tilde{h_{usl}}(x, t - 1) + i_{usl}(x, t)\\
            &\tilde{h_{msl}}(x, t)& &=& &\tilde{h_{msl}}(x, t - 1) + i_{msl}(x, t)

        \end{eqnarray}

    - Compute the runoff :math:`q_r`

    .. math::

        q_r(x, t) = p_n(x, t) - (i_{usl}(x, t) + i_{msl}(x, t))

    **Drainage**

    - Compute the soil moisture in the first two layers, :math:`w_{usl}` and :math:`w_{msl}`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &w_{usl}(x, t - 1)& &=& &\tilde{h_{usl}}(x, t - 1)c_{usl}(x)\\
            &w_{msl}(x, t - 1)& &=& &\tilde{h_{msl}}(x, t - 1)c_{msl}(x)

        \end{eqnarray}

    - Compute the drainage flux :math:`d_{umsl}` from the upper soil layer to medium soil layer

    .. math::

        d_{umsl}(x, t^*) = k_s(x) * \tilde{h_{usl}}(x, t - 1)^{p_{bc}}

    - Update the drainage flux :math:`d_{umsl}` according to under and over soil layer saturation

    .. math::

        d_{umsl}(x, t) = \min(d_{umsl}(x, t^*), \min(w_{usl}(x, t - 1), c_{msl}(x) - w_{msl}(x, t - 1)))

    - Update the first two layers reservoir states normalized, :math:`\tilde{h_{usl}}` and :math:`\tilde{h_{msl}}`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &\tilde{h_{usl}}(x, t)& &=& &\tilde{h_{usl}}(x, t - 1) - \frac{d_{umsl}(x, t)}{c_{usl}(x)}\\
            &\tilde{h_{msl}}(x, t)& &=& &\tilde{h_{msl}}(x, t - 1) + \frac{d_{umsl}(x, t)}{c_{msl}(x)}

        \end{eqnarray}

    .. note::
        
        The same approach is performed for drainage in the medium and bottom layers. Hence the three first steps are skiped for readability and the update of the reservoir states is directly written.

    - Update of the normalized reservoirs states, :math:`\tilde{h_{msl}}` and :math:`\tilde{h_{bsl}}`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &\tilde{h_{msl}}(x, t)& &=& &\tilde{h_{msl}}(x, t - 1) - \frac{d_{mbsl}(x, t)}{c_{msl}(x)}\\
            &\tilde{h_{bsl}}(x, t)& &=& &\tilde{h_{bsl}}(x, t - 1) + \frac{d_{mbsl}(x, t)}{c_{bsl}(x)}

        \end{eqnarray}

    **Baseflow**

    - Compute the baseflow :math:`q_b`

    .. math::
        :nowrap:

        \begin{eqnarray}

            q_b(x, t) =
            \begin{cases}

                \frac{d_{sm}(x)d_s(x)}{w_s(x)}\tilde{h_{bsl}}(x, t - 1) &\text{if} \; \tilde{h_{bsl}}(x, t - 1) > w_s(x) \\
                \frac{d_{sm}(x)d_s(x)}{w_s(x)}\tilde{h_{bsl}}(x, t - 1) + d_{sm}(x)\left(1 - \frac{d_s(x)}{w_s(x)}\right)\left(\frac{\tilde{h_{bsl}}(x, t - 1) - w_s(x)}{1 - w_s(x)}\right)^2 &\text{otherwise}
            
            \end{cases}

        \end{eqnarray}

    - Update the normalized bottom soil layer reservoir state :math:`\tilde{h_{bsl}}`

    .. math::

        \tilde{h_{bsl}}(x, t) = \tilde{h_{bsl}}(x, t - 1) - \frac{q_b(x, t)}{c_{bsl}(x)}

.. _math_num_documentation.forward_structure.routing_module:

Routing operator :math:`\mathcal{M}_{hy}`
-----------------------------------------

The following routing operators are grid-based and adapted to perform on the same grid as the snow and production operators. 
They take as input an 8-direction (D8) drainage plan :math:`\mathcal{D}_{\Omega}\left(x\right)` obtained through terrain elevation processing. 

For all the following models, the 2D flow routing problem over the spatial domain :math:`\Omega` reduces to a 1D problem by using the 
drainage plan :math:`\mathcal{D}_{\Omega}\left(x\right)`. The latest, for a given cell :math:`x\in\Omega` defines 1 to 7 upstream cells which 
surface discharge can inflow the current cell :math:`x` - each cell has a unique downstream cell.


.. image:: ../_static/routing_module.svg
    :align: center
    :width: 300

.. _math_num_documentation.forward_structure.routing_module.lag0:

.. dropdown:: Instantaneous Routing (lag0)
    :animate: fade-in-slide-down

    This routing operator is a simple aggregation of upstream discharge to downstream following the drainage plan. It can be expressed as follows:

    .. math::

        Q(x, t) = f\left(Q(x', t), q_{t}(x, t)\right),\;\forall x'\in \Omega_x

    with :math:`Q` the surface discharge, :math:`q_t` the elemental discharge and :math:`\Omega_x` a 2D spatial domain that corresponds to all upstream cells
    flowing into cell :math:`x`, i.e. the whole upstream catchment. Note that :math:`\Omega_x` is a subset of :math:`\Omega`, :math:`\Omega_x\subset\Omega` and for the most upstream cells, 
    :math:`\Omega_x=\emptyset`.

    .. note::

        Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
        
        - Surface discharge: :math:`Q`
        - Internal fluxes: :math:`\{q_{t}\}\in\boldsymbol{q}`

    The function :math:`f` is resolved numerically as follows:

    **Upstream discharge**

    - Compute the upstream discharge :math:`q_{up}`

    .. math::
        :nowrap:

        \begin{eqnarray}

            q_{up}(x, t) = 
            \begin{cases}

                0 &\text{if} \; \Omega_x = \emptyset \\
                \sum_{k\in\Omega_x} Q(k, t) &\text{otherwise}

            \end{cases}

        \end{eqnarray}

    **Surface discharge**

    - Compute the surface discharge :math:`Q`

    .. math::

        Q(x, t) = q_{up}(x, t) + \alpha(x) q_t(x, t)

    with :math:`\alpha` a conversion factor from :math:`mm.\Delta t^{-1}` to :math:`m^3.s^{-1}` for a single cell.

.. _math_num_documentation.forward_structure.routing_module.lr:

.. dropdown:: Linear Reservoir (lr)
    :animate: fade-in-slide-down

    This routing operator is using a linear reservoir to rout upstream discharge to downstream following the drainage plan. It can be expressed as follows:

    .. math::

        Q(x, t) = f\left(Q(x', t), q_{t}(x, t), l_{lr}(x), h_{lr}(x, t)\right),\;\forall x'\in \Omega_x

    with :math:`Q` the surface discharge, :math:`q_t` the elemental discharge, :math:`l_{lr}` the routing lag time, 
    :math:`h_{lr}` the state of the routing reservoir and :math:`\Omega_x` a 2D spatial domain that corresponds to all upstream cells
    flowing into cell :math:`x`. Note that :math:`\Omega_x` is a subset of :math:`\Omega`, :math:`\Omega_x\subset\Omega` and for the most upstream cells, 
    :math:`\Omega_x=\emptyset`.

    .. note::

        Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`
        
        - Surface discharge: :math:`Q`
        - Internal fluxes: :math:`\{q_{t}\}\in\boldsymbol{q}`
        - Parameters: :math:`\{l_{lr}\}\in\boldsymbol{\theta}`
        - Normalized states: :math:`\{\tilde{h_{lr}}\}`, where :math:`\tilde{h_{lr}} = \frac{h_{lr}}{l_{lr}}`, with states :math:`\{h_{lr}\} \in \boldsymbol{h}`

    The function :math:`f` is resolved numerically as follows:

    **Upstream discharge**

    Same as ``lag0`` upstream discharge, see :ref:`LAG0 Upstream Discharge <math_num_documentation.forward_structure.routing_module.lag0>`.

    **Surface discharge**

    - Update the normalized routing reservoir state :math:`\tilde{h_{lr}}`

    .. math::

        \tilde{h_{lr}}(x, t^*) = \tilde{h_{lr}}(x, t) + \frac{1}{\beta(x)} q_{up}(x, t)

    with :math:`\beta` a conversion factor from :math:`mm.\Delta t^{-1}` to :math:`m^3.s^{-1}` for the whole upstream domain :math:`\Omega_x`.

    - Compute the routed discharge :math:`q_{rt}`

    .. math::

        q_{rt}(x, t) = \tilde{h_{lr}}(x, t^*) \left(1 - \exp\left(\frac{-\Delta t}{60\times l_{lr}}\right)\right)

    - Update the normalized routing reservoir state :math:`\tilde{h_{lr}}`

    .. math::

        \tilde{h_{lr}}(x, t) = \tilde{h_{lr}}(x, t^*) - q_{rt}(x, t)

    - Compute the surface discharge :math:`Q`

    .. math::

        Q(x, t) = \beta(x)q_{rt}(x, t) + \alpha(x)q_t(x, t)

    with :math:`\alpha` a conversion factor from from :math:`mm.\Delta t^{-1}` to :math:`m^3.s^{-1}` for a single cell.

.. _math_num_documentation.forward_structure.routing_module.kw:

.. dropdown:: Kinematic Wave (kw)
    :animate: fade-in-slide-down

    This routing operator is based on a conceptual 1D kinematic wave model that is numerically solved with a linearized implicit numerical scheme :cite:p:`ChowAppliedhydrology`. This is applicable given the drainage plan :math:`\mathcal{D}_{\Omega}\left(x\right)` that enables reducing the routing problem to 1D. 

    The kinematic wave model is a simplification of 1D Saint-Venant hydraulic model. First the mass equation writes:

    .. math:: 
        :name: math_num_documentation.forward_structure.forward_problem_mass_KW

        \partial_{t}A+\partial_{x}Q =q
        
    with :math:`\partial_{\square}` denoting the partial derivation either in time or space, :math:`A` the cross sectional flow area, :math:`Q` the flow discharge and :math:`q` the lateral inflows. 

    Assuming that the momentum equation reduces to

    .. math:: 
        :name: math_num_documentation.forward_structure.forward_problem_momentum_KW
        
        S_0=S_f
        
    with :math:`S_0` the bottom slope and :math:`S_f` the friction slope - i.e. a locally uniform flow with energy grade line parallel to the channel bottom. 
    This momentum equation can be expressed in the following form, as described by :cite:t:`ChowAppliedhydrology`

    .. math::
        :name: math_num_documentation.forward_structure.conceptual_A_of_Q
        
        A=a_{kw} Q ^{b_{kw}}

    with :math:`a_{kw}` and :math:`b_{kw}` two constants to be estimated - that can also be written using Manning friction law.

    Injecting the momentum parameterization of :ref:`Eq. 3 <math_num_documentation.forward_structure.conceptual_A_of_Q>` into mass equation :ref:`Eq. 1 <math_num_documentation.forward_structure.forward_problem_mass_KW>` 
    leads to the following one equation kinematic wave model :cite:p:`ChowAppliedhydrology`:

    .. math:: 
        :name: math_num_documentation.forward_structure.oneEq_KW_conceptual

        \partial_{x}Q+a_{kw}b_{kw} Q^{b_{kw}-1}\partial_{t}Q=q

    .. hint::

        Helpful link about kinematic wave:

        - `Numerical Solution <https://wecivilengineers.files.wordpress.com/2017/10/applied-hydrology-ven-te-chow.pdf>`__ (page 294, section 9.6)

    The solution of this equation can written as:

    .. math::

        Q(x, t) = f\left(Q(x', t'), q_{t}(x, t'), \left[a_{kw}, b_{kw}\right](x)\right),\;\forall (x', t') \in \Omega_x\times[t-1, t]

    with :math:`Q` the surface discharge, :math:`q_t` the elemental discharge, :math:`a_{kw}` the alpha kinematic wave parameter, 
    :math:`b_{kw}` the beta kinematic wave parameter and :math:`\Omega_x` a 2D spatial domain that corresponds to all upstream cells
    flowing into cell :math:`x`. Note that :math:`\Omega_x` is a subset of :math:`\Omega`, :math:`\Omega_x\subset\Omega` and for the most upstream cells, 
    :math:`\Omega_x=\emptyset`.

    .. note::

        Linking with the forward problem equation :eq:`math_num_documentation.forward_inverse_problem.forward_problem_M_1`

        - Surface discharge: :math:`Q`
        - Internal fluxes: :math:`\{q_{t}\}\in\boldsymbol{q}`
        - Parameters: :math:`\{a_{kw}, b_{kw}\}\in\boldsymbol{\theta}`

    For the sake of clarity, the following variables are renamed for this section and the finite difference numerical scheme writing:

    .. list-table:: Renamed variables
        :widths: 25 25
        :header-rows: 1

        * - Before
          - After
        * - :math:`Q(x, t)`
          - :math:`Q_i^j`
        * - :math:`Q(x, t - 1)`
          - :math:`Q_{i}^{j-1}`
        * - :math:`q_t(x, t)`
          - :math:`q_{i}^{j}`
        * - :math:`q_t(x, t - 1)`
          - :math:`q_{i}^{j-1}`

    The function :math:`f` is resolved numerically as follows:

    **Upstream discharge**

    Same as ``lag0`` upstream discharge, see :ref:`LAG0 Upstream Discharge <math_num_documentation.forward_structure.routing_module.lag0>`.

    .. note::

        :math:`q_{up}` is denoted here :math:`Q_{i-1}^{j}`

    **Surface discharge**

    - Compute the intermediate variables :math:`d_1` and :math:`d_2`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &d_1& &=& &\frac{\Delta t}{\Delta x}\\
            &d_2& &=& &a_{kw} b_{kw} \left(\frac{\left(Q_i^{j-1} + Q_{i-1}^j\right)}{2}\right)^{b_{kw} - 1}

        \end{eqnarray}

    - Compute the intermediate variables :math:`n_1`, :math:`n_2` and :math:`n_3`

    .. math::
        :nowrap:

        \begin{eqnarray}

            &n_1& &=& &d_1 Q_{i-1}^j\\
            &n_2& &=& &d_2 Q_{i}^{j-1}\\
            &n_3& &=& &d_1 \frac{\left(q_i^{j-1} + q_{i}^{j}\right)}{2}

        \end{eqnarray}

    - Compute the surface discharge :math:`Q_i^j`

    .. math::

        Q_i^j = Q(x, t) = \frac{n_1 + n_2 + n_3}{d_1 + d_2}
