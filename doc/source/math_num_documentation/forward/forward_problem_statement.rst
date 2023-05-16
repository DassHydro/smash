.. _math_num_documentation.forward.forward_problem_statement:

=========================
Forward problem statement
=========================

The **computational domain** (catchment) :math:`\Omega \in \mathbb{R}^2` is a 2D spatial domain and :math:`t > 0` denotes the physical time. A lattice :math:`\mathcal{T}_{\Omega}` covers :math:`\Omega` and the number of active cells within a catchment :math:`\Omega` is denoted :math:`N_x`. Let
:math:`\mathcal{D}_\Omega(x)\forall x \in \Omega` be the drainage plan obtained from terrain elevation processing considering river network.

.. DEFINIR propre mm support spatial pr drainage plan, descriptors, all fields...

The **hydrological model** is a dynamic operator :math:`\mathcal{M}` mapping observed input fields of rainfall, evapotranspiration and eventually snow and temperature :math:`\boldsymbol{P}(x, t'), \; \boldsymbol{E}(x, t'), \; \boldsymbol{S}(x, t'), \; \boldsymbol{T}(x, t'), \; \forall (x, t') \in \Omega \times [0, t]` onto discharge field :math:`Q(x, t)` such that:

.. math::
   :name: eq:forward-model
   
      Q\left(x,t\right)=\mathcal{M}\left[\boldsymbol{P}\left(x,t'\right),\boldsymbol{E}\left(x,t'\right),\boldsymbol{h}\left(x,0\right),\boldsymbol{\theta}\left(x\right)\right]\,\forall x\in\Omega, t'\in\left[0,t\right]
    
with :math:`\boldsymbol{h}(x, t)` the :math:`N_s`-dimensional vector of model states 2D fields and :math:`\boldsymbol{\theta}` the :math:`N_p`-dimensional vector of model parameters 2D fields.

Note that a pre-regionalization function :math:`\mathcal{F}_{R}` can be considered in the forward model to link model parameters to physiographic descriptors :math:`\boldsymbol{D}` (see section :ref:`Regionalization operators <math_num_documentation.forward.regionalization_operators>`) such that:

.. math::
   :name: eq:regio-mapping
   
      \boldsymbol{\theta}(x)=\mathcal{F}_{R}(\boldsymbol{D}(x),\boldsymbol{\rho}(x)),\,\forall\left(x\right)\in\Omega
   
with :math:`\boldsymbol{\rho}` the tunable parameters that is regionalization control vector. In that case the forward model is a composed function :math:`\mathcal{M}\left(\mathcal{F}_{R}\left(\rho\right)\right)` depending on :math:`\boldsymbol{\rho}`.

Remark that all operators of the forward model can contain trainable neural networks.
