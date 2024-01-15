.. _math_num_documentation.forward_inverse_problem:

=========================
Forward / Inverse Problem
=========================

Forward problem
---------------

Let :math:`\Omega\subset\mathbb{R}^{2}` denote a 2D spatial domain with :math:`x` the spatial coordinate and :math:`t\in\left]0,T\right]` the physical time.

Hydrological model
******************

The spatially distributed hydrological model is a dynamic operator :math:`\mathcal{M}` projecting fields of atmospheric forcings :math:`\mathcal{\boldsymbol{I}}`,
catchment physiographic descriptors :math:`\boldsymbol{\mathcal{D}}` onto surface discharge :math:`Q`, model states :math:`\boldsymbol{h}`, and internal fluxes :math:`\boldsymbol{q}` such that:

.. math::
    :name: math_num_documentation.forward_inverse_problem.forward_problem_M_1

    \boldsymbol{U}(x,t)=(Q,\boldsymbol{h},\boldsymbol{q})(x,t)=\mathcal{M}\left(\left[\mathcal{\boldsymbol{I}},\boldsymbol{\mathcal{D}}\right](x,t);\left[\boldsymbol{\theta},\boldsymbol{h}_{0}\right](x)\right)

with :math:`\boldsymbol{U}(x,t)` the modeled state-flux variables, :math:`\boldsymbol{\theta}` the parameters and :math:`\boldsymbol{h}_{0}` the initial states.

.. note::

    - Surface discharge: :math:`Q\in\Omega\times\left]0, T\right]`,

    - States: :math:`\boldsymbol{h}=\left(h_{1}(x,t),...,h_{N_{h}}(x,t)\right)\in\Omega^{N_{h}}\times\left]0, T\right]`

    - Internal fluxes: :math:`\boldsymbol{q}=\left(q_{1}(x,t),...,q_{N_{q}}(x,t)\right)\in\Omega^{N_{q}}\times\left]0, T\right]`

    - Atmospheric forcings: :math:`\mathcal{\boldsymbol{I}}=\left(\mathcal{I}_{1}(x,t),...,\mathcal{I}_{N_{\mathcal{I}}}(x,t)\right)\in\Omega^{N_{\mathcal{I}}}\times\left]0, T\right]`

    - Physiographic descriptors: :math:`\mathcal{\boldsymbol{D}}=\left(\mathcal{D}_{1}(x,t),...,\mathcal{D}_{N_{\mathcal{D}}}(x,t)\right)\in\Omega^{N_{\mathcal{D}}}\times\left]0, T\right]`

    - Parameters: :math:`\boldsymbol{\theta}=\left(\theta_{1}(x),...,\theta_{N_{\theta}}(x)\right)\in\Omega^{N_{\theta}}\times\left]0, T\right]`

    - Initial states: :math:`\boldsymbol{h}_{0}=\boldsymbol{h}(x,t=0)`.

Operators composition
*********************

Note that the operator :math:`\mathcal{M}` can be a composite function containing, differentiable operators for vertical and lateral transfert processes within each cell :math:`x\in\Omega`, 
and routing operator from cells to cells following a flow direction map, plus deep neural networks enabling learnable process parameterization-regionalization capabilities as described later.

.. _math_num_documentation.forward_inverse_problem.mapping:

Mapping
*******

The spatio-temporal fields of model parameters and initial states can be constrained with spatialization rules (e.g., control spatial reduction into patches), or even explained by physiographic descriptors 
:math:`\boldsymbol{\mathcal{D}}`. This is achieved via the operator :math:`\phi: \left(\boldsymbol{\theta}(x),\boldsymbol{h}_{0}(x)\right)=\phi\left(\boldsymbol{\mathcal{D}}(x,t),\boldsymbol{\rho}\right)`
with :math:`\boldsymbol{\rho}` the control vector that can be optimized.

Finally, replacing in :ref:`Eq. 1 <math_num_documentation.forward_inverse_problem.forward_problem_M_1>` the parameters and initial states by the :math:`\phi` operator, the differentiable forward model writes as: 

.. math::
    :name: math_num_documentation.forward_inverse_problem.forward_problem_M_2

    \mathcal{M}\left(\left[\mathcal{\boldsymbol{I}},\mathcal{\boldsymbol{D}}\right](x,t);\phi\left(\boldsymbol{\mathcal{D}}(x,t),\boldsymbol{\rho}\right)\right)

Inverse problem
---------------

Consider the following generic cost function composed of and observation term :math:`J_{obs}` and a regularization term :math:`J_{reg}` weighted by :math:`\alpha\geq0`:


.. _math_num_documentation.forward_inverse_problem.cost_function:

Cost function
*************

.. math::
    :name: math_num_documentation.forward_inverse_problem.inverse_problem_J

    J=J_{obs}+\alpha J_{reg}

Observation term
****************

The modeled states variables :math:`\boldsymbol{U}(x,t)=(Q,\boldsymbol{h},\boldsymbol{q})(x,t)` are observed in a vector 
:math:`\boldsymbol{Y}=H\left[\mathcal{M}(\boldsymbol{\rho})\right]\in\mathcal{Y}` with :math:`H:\mathcal{X}\mapsto\mathcal{Y}` 
the observation operator from state space :math:`\mathcal{X}` to observation space :math:`\mathcal{Y}`.

Given observations :math:`\boldsymbol{Y}^{*}(x^{*},t^{*})\in\mathcal{Y}` of hydrological responses over the domain :math:`\Omega\times]0,T]`, 
the model misfit to observations is measured through the observation cost function:

.. math::

    J_{obs}=\frac{1}{2}\left\Vert \boldsymbol{Y}-\boldsymbol{Y}^{*}\right\Vert _{O}^{2}

.. math::
    :name: math_num_documentation.forward_inverse_problem.inverse_problem_Jobs

    J_{obs}\left(\boldsymbol{\rho}\right)=\frac{1}{2}\left\Vert H\left[\mathcal{M}(\boldsymbol{\rho})\right]-\boldsymbol{Y^{*}}\right\Vert _{O}^{2}

with :math:`O` the observation error covariance matrix and the euclidian norm :math:`\left\Vert X\right\Vert {O}^{2}=X^{T}OX` 

Regularization term
*******************

The regularization term is for example a Thikhonov regularization that only involves the control :math:`\boldsymbol{\rho}` and its background value :math:`\boldsymbol{\rho}^*` from which optimization is started.

Optimization
************

The optimization problem minimizing the misfit :math:`J` to observations writes as:

.. math::
    :name: math_num_documentation.forward_inverse_problem.inverse_problem_optimization

    \boldsymbol{\hat{\rho}}=\underset{\mathrm{\boldsymbol{\rho}}}{\text{argmin}}J

This problem can be tackled with optimization algorithms adapted to high dimensional problems (L-BFGS-B or machine learning optimizers (e.g., Adam) (TODO link to Math / Num)) that require the gradient :math:`\nabla_{\boldsymbol{\rho}}J` 
of the cost function to the sought parameters :math:`\boldsymbol{\rho}`. The computation of the cost gradient :math:`\nabla_{\boldsymbol{\rho}}J` relies on the composed adjoint model :math:`\Lambda` 
that is derived by automatic differenciation of the forward model, using the Tapenade software :cite:p:`hascoet2013tapenade`.

.. note::

    Following this general definition of the inverse problem, multiple definition of observation cost function, regularization as well as mappings affecting the control are possible with `smash`
    and detailled after as well as the optimization algorithms taylored to solve them.

