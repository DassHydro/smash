.. _math_num_documentation.mapping:

=======
Mapping
=======

This section follows on from the :ref:`math_num_documentation.forward_inverse_problem.mapping` section introduced in the forward problem.
The aim of this section is to present all the :math:`\phi` operators that can be used to map a control vector :math:`\boldsymbol{\rho}` 
to the parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h_0}`. 

Each mapping section contains two sub-sections:

- Initialization
- Control vector to parameters and initial states

The first section defines how the control vector :math:`\boldsymbol{\rho}` is initialized to start the optimization 
and the second section defines how to map the control vector :math:`\boldsymbol{\rho}` to the parameters :math:`\boldsymbol{\theta}` 
and initial states :math:`\boldsymbol{h_0}`.

.. note::

    The initialization method cannot be set up as the inverse function of :math:`\phi` (except for distributed mapping) 
    because the :math:`\phi` operator is non-bijective. An arbitrary pseudo inverse function is therefore used for each mapping.

Uniform
-------

Initialization
**************

The control vector :math:`\rho_k` initialized by taking the spatial average of each parameter :math:`\theta_k` and initial state :math:`h_{0_k}`.

.. math::
    :nowrap:

    \begin{eqnarray}

        \phi^{-1}: \; &\Omega \mapsto \; &\mathbb{R}\\
                      &\theta_k \mapsto &\rho_k = \frac{1}{\lvert\Omega\rvert}\sum_{x\in\Omega}\theta_k(x) \;\;\; &\forall (k, x) \in [1 .. N_{\theta}] \times \Omega\\
                      &h_{0_k} \mapsto &\rho_k = \frac{1}{\lvert\Omega\rvert}\sum_{x\in\Omega}h_{0_k}(x) \;\;\; &\forall (k, x) \in [1 .. N_{h}] \times \Omega

    \end{eqnarray}

Control vector to parameters and initial states
***********************************************

The control vector :math:`\rho_k` is mapped into spatially uniform parameters :math:`\theta_k` and initial states :math:`h_{0_k}`.

.. math::
    :nowrap:

    \begin{eqnarray}

        \phi: \; &\mathbb{R} \mapsto \; &\Omega\\
                 &\rho_k \mapsto &\theta_k = (\rho_k, \; .. \; \rho_k) \;\;\; &\forall k \in [1 .. N_{\theta}]\\
                 &\rho_k \mapsto &h_{0_k} = (\rho_k, \; .. \; \rho_k) \;\;\; \;\;\; &\forall k \in [1 .. N_{h}]

    \end{eqnarray}

Distributed
-----------

Initialization
**************

The control vector :math:`\rho_k` is initialized by taking each value of parameter :math:`\theta_k` and initial state :math:`h_{0_k}`.

.. math::
    :nowrap:

    \begin{eqnarray}

        \phi^{-1}: \; &\Omega \mapsto \; &\Omega\\
                      &\theta_k \mapsto &\rho_k = \theta_k \;\;\; &\forall k \in [1 .. N_{\theta}]\\
                      &h_{0_k} \mapsto &\rho_k = h_{0_k} \;\;\; &\forall k \in [1 .. N_{h}]

    \end{eqnarray}

Control vector to parameters and initial states
***********************************************

The control vector :math:`\rho_k` is mapped into spatially distributed parameters :math:`\theta_k` and initial states :math:`h_{0_k}`.

.. math::
    :nowrap:

    \begin{eqnarray}

        \phi: \; &\Omega \mapsto \; &\Omega\\
                 &\rho_k \mapsto &\theta_k = \rho_k \;\;\; &\forall k \in [1 .. N_{\theta}]\\
                 &\rho_k \mapsto &h_{0_k} = \rho_k \;\;\; &\forall k \in [1 .. N_{h}]

    \end{eqnarray}

.. _math_num_documentation.mapping.multi_linear:

Multi-linear
------------

Initialization
**************

The control vector :math:`\rho_k` is initialized by taking the scaled inverse sigmoide :math:`s_k^{-1}` of the spatial average 
of each parameter :math:`\theta_k` and initial state :math:`h_{0_k}` for the intercept of the multivariate linear regression and 
setting 0 for all physiographic descriptor related coefficients.

.. note::

    The intercept of the multivariate linear regression is considered to be the first element of :math:`\rho_k` 
    for each parameter :math:`\theta_k` and initial state :math:`h_{0_k}`.

.. math::
    :nowrap:

    \begin{eqnarray}

        \phi^{-1}: \; &\Omega \mapsto \; &\mathbb{R}^{N_\mathcal{D} + 1}\\
                      &\theta_k \mapsto &\rho_k = \left(s_k^{-1}\left(\frac{1}{\lvert\Omega\rvert}\sum_{x\in\Omega}\theta_k(x)\right), 0, \; .. \; 0\right) \;\;\; &\forall (k, x) \in [1 .. N_{\theta}] \times \Omega\\
                      &h_{0_k} \mapsto &\rho_k = \left(s_k^{-1}\left(\frac{1}{\lvert\Omega\rvert}\sum_{x\in\Omega}h_{0_k}(x)\right), 0, \; .. \; 0\right) \;\;\; &\forall (k, x) \in [1 .. N_{h}] \times \Omega\\

    \end{eqnarray}

The scaled inverse sigmoide :math:`s_k^{-1}` is expressed as follows:

.. math::
    :nowrap:

    \begin{eqnarray}

        s_k^{-1}: \; ]l_k, u_k[& \mapsto \; &\mathbb{R}\\
                     x& \mapsto &\ln\left(\frac{x - l_k}{u_k - x}\right)

    \end{eqnarray}

with :math:`l_k` and :math:`u_k` the bound constraints on the parameter :math:`\theta_k` or initial state :math:`h_{0_k}` such that 
:math:`l_k < [\theta_k, h_{0_k}](x) < u_k, \forall x \in \Omega`

Control vector to parameters and initial states
***********************************************

The control vector :math:`\rho_k` is mapped into spatially distributed parameters :math:`\theta_k` and initial states :math:`h_{0_k}` 
using a scaled sigmoide :math:`s_{k}` multivariate linear regression.

.. math::
    :nowrap:

    \begin{eqnarray}

        \phi: \; \mathbb{R}^{N_\mathcal{D} + 1}& \mapsto \; &\Omega\\
                 \rho_k& \mapsto &\theta_k = s_k\left(\rho_{k,1} + \sum_{d=2}^{N_{\mathcal{D}} + 1} \rho_{k,d}\mathcal{D}_d \right) \;\;\; &\forall k \in [1 .. N_{\theta}]\\
                 \rho_k& \mapsto &h_{0_k} = s_k\left(\rho_{k,1} + \sum_{d=2}^{N_{\mathcal{D}} + 1} \rho_{k,d}\mathcal{D}_d \right) \;\;\; &\forall k \in [1 .. N_{h}]

    \end{eqnarray}

with :math:`\mathcal{D}` the physiographic descriptor.

The scaled sigmoide :math:`s_k` is expressed as follows:

.. math::
    :nowrap:

    \begin{eqnarray}

        s_k: \; \mathbb{R}& \mapsto \; &]l_k, u_k[\\
                x& \mapsto &l_{k} + \frac{u_{k}-l_{k}}{1 + e^{- x}}

    \end{eqnarray}

with :math:`l_k` and :math:`u_k` the bound constraints on the parameter :math:`\theta_k` or initial state :math:`h_{0_k}` such that 
:math:`l_k < [\theta_k, h_{0_k}](x) < u_k, \forall x \in \Omega`

Multi-polynomial
----------------

Initialization
**************

The control vector :math:`\rho_k` is initialized by taking the scaled inverse sigmoide :math:`s_{k}^{-1}` of the spatial average of 
each parameter :math:`\theta_k` and initial state :math:`h_{0_k}` for the intercept of the multivariate polynomial regression, 
setting 0 for all physiographic descriptor related coefficients and 1 for exponents.

.. note::

    The intercept of the multivariate polynomial regression is considered to be the first element of :math:`\rho_k` 
    for each parameter :math:`\theta_k` and initial state :math:`h_{0_k}` and next come the coefficient and exponent pairs for 
    each physiographic descriptor.

.. math::
    :nowrap:

    \begin{eqnarray}

        \phi^{-1}: \; &\Omega \mapsto \; &\mathbb{R}^{2N_\mathcal{D} + 1}\\
                      &\theta_k \mapsto &\rho_k = \left(s_k^{-1}\left(\frac{1}{\lvert\Omega\rvert}\sum_{x\in\Omega}\theta_k(x)\right), 0, 1, \; .. \; 0, 1\right) \;\;\; &\forall (k, x) \in [1 .. N_{\theta}] \times \Omega\\
                      &h_{0_k} \mapsto &\rho_k = \left(s_k^{-1}\left(\frac{1}{\lvert\Omega\rvert}\sum_{x\in\Omega}h_{0_k}(x)\right), 0, 1, \; .. \; 0, 1\right) \;\;\; &\forall (k, x) \in [1 .. N_{h}] \times \Omega\\

    \end{eqnarray}

The scaled inverse sigmoide :math:`s_k^{-1}` is expressed in the :ref:`Multi-linear <math_num_documentation.mapping.multi_linear>` section.

Control vector to parameters and initial states
***********************************************

The control vector :math:`\rho_k` is mapped into spatially distributed parameters :math:`\theta_k` and initial states :math:`h_{0_k}` using a 
scaled sigmoide :math:`s_{k}` multivariate polynomial regression.

.. math::
    :nowrap:

    \begin{eqnarray}

        \phi: \; \mathbb{R}^{2N_\mathcal{D} + 1}& \mapsto \; &\Omega\\
                 \rho_k& \mapsto &\theta_k = s_k\left(\rho_{k,1} + \sum_{d=2}^{N_{\mathcal{D}} + 1} \rho_{k,d^*}\left(\mathcal{D}_d\right)^{\rho_{k, d^* + 1}} \right) \;\;\; &\forall k \in [1 .. N_{\theta}]\\
                 \rho_k& \mapsto &h_{0_k} = s_k\left(\rho_{k,1} + \sum_{d=2}^{N_{\mathcal{D}} + 1} \rho_{k,d^*}\left(\mathcal{D}_d\right)^{\rho_{k, d^* + 1}} \right) \;\;\; &\forall k \in [1 .. N_{h}]\\
    \end{eqnarray}

with :math:`\mathcal{D}` the physiographic descriptor and :math:`d^*=2(d-1)`

The scaled sigmoide :math:`s_k` is expressed in the :ref:`Multi-linear <math_num_documentation.mapping.multi_linear>` section.

ANN
---

Initialization
**************

In this case, the control vector :math:`\boldsymbol{\rho}`, representing the weights and biases of the ANN, is randomly initialized using one of the following methods:

- Zero initialization: :math:`\boldsymbol{\rho} = \mathbf{0}`
- Default uniform initialization: :math:`\boldsymbol{\rho} \sim \mathcal{U}\left(-\sqrt{\frac{1}{n_{in}}}, \sqrt{\frac{1}{n_{in}}}\right)`
- He uniform initialization: :math:`\boldsymbol{\rho} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)`
- Glorot uniform initialization: :math:`\boldsymbol{\rho} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)`
- Default normal initialization: :math:`\boldsymbol{\rho} \sim \mathcal{G}(0, 0.01)`
- He normal initialization: :math:`\boldsymbol{\rho} \sim \mathcal{G}(0, \sqrt{\frac{2}{n_{in}}})`
- Glorot normal initialization: :math:`\boldsymbol{\rho} \sim \mathcal{G}(0, \sqrt{\frac{2}{n_{in} + n_{out}}})`

with :math:`n_{in}` and :math:`n_{out}` denote the number of neurons in the input and output layers, respectively; :math:`\mathcal{U}` and :math:`\mathcal{G}` denote the uniform and Gaussian distributions, respectively.

Control vector to parameters and initial states
***********************************************

The control vector :math:`\boldsymbol{\rho}` is mapped into spatially distributed parameters :math:`\boldsymbol{\theta}` and 
initial states :math:`\boldsymbol{h}` using an artificial neural network (ANN) denoted :math:`\mathcal{N}`. 
It consists of a multilayer perceptron aiming to learn the physiographic descriptors :math:`\boldsymbol{\mathcal{D}}` 
to parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h}` mapping.

.. math::
    :nowrap:

    \begin{eqnarray}

        \phi: \; \mathbb{R}^{N_{W} + N_{b}}& \mapsto \; &\Omega^{N_\theta + N_h}\\
                \boldsymbol{\rho}& \mapsto \; &[\boldsymbol{\theta}, \boldsymbol{h}] = \mathcal{N}\left(\boldsymbol{\mathcal{D}}, \boldsymbol{\rho}[\boldsymbol{W}, \boldsymbol{b}] \right)
                 
    \end{eqnarray}

where :math:`\boldsymbol{W}` and :math:`\boldsymbol{b}` are respectively weights and biases of the neural network composed of 
:math:`N_L` dense layers. Note that an output layer consisting in a transformation based on the sigmoid function enables to impose 
:math:`l_k < [\theta_k, h_{0_k}](x) < u_k, \forall x \in \Omega`, i.e. bounds constrains on parameters and initial states.

The following figure illustrates the architecture of the ANN with three hidden layers, followed by the ReLU activation function, 
and an output layer that uses the Sigmoid activation function in combination with a scaling function. In this particular case,
we have :math:`N_{\mathcal{D}} = 7` and :math:`N_{\theta} + N_{h} = 4`.

.. image:: ../_static/FCNN.png
    :width: 750
    :align: center
