.. _math_num_documentation.mapping:

=======
Mapping
=======

This section, follows on from the :ref:`math_num_documentation.forward_inverse_problem.mapping` section and the general definition of the mapping operator :math:`\phi` (:ref:`Eq. 3 <math_num_documentation.forward_inverse_problem.mapping_general>`) that is a component of the forward model. 
A mapping :math:`\phi` defines how the optimizable control vector :math:`\boldsymbol{\rho}`, and optionnally physical descriptors :math:`\boldsymbol{\mathcal{D}}`, are mapped onto the conceptual model parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h}_0`. 
This section provides a detailled definition of all the variants available for defining :math:`\phi` in the forward model.

For each mapping variant two sub-sections successively detail:

- The mapping formulation, i.e. how the optimizable control vector :math:`\boldsymbol{\rho}`, belonging to the control space :math:`\mathcal{K}`, is mapped onto parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h}_0`.
- The control vector initialization, i.e. how the prior/first guess/starting point :math:`\boldsymbol{\rho}^*` is set for starting optimization - which depends on the optimization algorithm.

.. note::
      
      A mapping function :math:`\phi` is part of the forward model :math:`\mathcal{M}` and the corresponding forward code is differentiated.


Spatially Uniform
-----------------

Mapping formulation
*******************

The control vector :math:`\boldsymbol{\rho}` is mapped in order to impose spatially uniform parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h_0}`, such that for :math:`k \in  [1 .. N_{\theta}]` conceptual parameters and for :math:`s \in  [1 .. N_{h}]` states and :math:`\forall x\in\Omega`: 

.. math::
    
    \phi:\left(\boldsymbol{\theta},\boldsymbol{h}_{0}\right)(x)=\overline{\left(\boldsymbol{\theta},\boldsymbol{h}_{0}\right)}=\left(\boldsymbol{\rho}_{k};\boldsymbol{\rho}_{N_{\theta}+s}\right)
    
where ":math:`\;\overline{.}\;`" denotes a constant value over the spatial domain :math:`\Omega`.

First guess
***********

The first guess :math:`\boldsymbol{\rho}^*` on the control vector :math:`\boldsymbol{\rho}` to optimize is defined accordingly with spatially constant values such that for :math:`k \in  [1 .. N_{\theta}]` conceptual parameters and for :math:`s \in  [1 .. N_{h}]` states and :math:`\forall x\in\Omega`: 


.. math::
    
    \boldsymbol{\rho}^{*}(x)=\left(\boldsymbol{\theta}^{*},\boldsymbol{h}_{0}^{*}\right)(x)=\overline{\left(\boldsymbol{\theta}^{*},\boldsymbol{h}_{0}^{*}\right)}=\left(\boldsymbol{\rho}_{k}^{*};\boldsymbol{\rho}_{N_{\theta}+s}^{*}\right)

.. note::

       A spatially uniform control of the spatially distributed hydrological model generally leads to under-parameterized inverse problems from spatially sparse discharge data. One would need to relax this spatial constrain to improve the model predictive performances that can be reached in optimization.

Spatially Distributed
---------------------


Mapping formulation
*******************

The control vector :math:`\boldsymbol{\rho}` is mapped in order to impose spatially distributed parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h_0}`, such that for :math:`k \in  [1 .. N_{\theta}]` conceptual parameters and for :math:`s \in  [1 .. N_{h}]` states and :math:`\forall x\in\Omega`:  

.. math:: 

       \phi:\left(\boldsymbol{\theta},\boldsymbol{h}_{0}\right)(x)=\left(\boldsymbol{\rho}_{k};\boldsymbol{\rho}_{N_{\theta}+s}\right)(x)


First guess
***********

The first guess :math:`\boldsymbol{\rho}^*` on the control vector :math:`\boldsymbol{\rho}` is defined accordingly with spatially distributed values such that for :math:`k \in  [1 .. N_{\theta}]` conceptual parameters and for :math:`s \in  [1 .. N_{h}]` states and :math:`\forall x\in\Omega`: 


.. math::
    
    \boldsymbol{\rho}^{*}(x)=\left(\boldsymbol{\theta}^{*},\boldsymbol{h}_{0}^{*}\right)(x)=\left(\boldsymbol{\rho}_{k}^{*};\boldsymbol{\rho}_{N_{\theta}+s}^{*}\right)(x)

.. note::

       A spatially uniform control of the spatially distributed hydrological model generally leads to over-parameterized inverse problems from spatially sparse discharge data. One would need to introduce spatial constrains on parameters fields for improving optimization meaningfulness.
    
.. _math_num_documentation.mapping.multi_linear:

Multi-linear
------------

This mapping enables to use physical despcriptors :math:`\boldsymbol{\mathcal{D}}` to both spatially constrain and explain the parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h_0}` of the conceptual model.

Mapping formulation
*******************

The control vector :math:`\boldsymbol{\rho}` and the physical despcriptors :math:`\boldsymbol{\mathcal{D}}=\left(D_{d}(x)\right),\,d\in[1..N_{D}]` 
is mapped in order to impose spatially distributed parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h_0}`, such that :math:`\forall x\in\Omega`:  

.. math::

     \phi:\begin{cases}
     
     \theta_{k}(x) & =s_{k}\left(\rho_{k,0}+\sum_{d=1}^{N_{D}}\rho_{k,d}D_{d}(x)\right)\;\;\;,\,k\in[1..N_{\theta}]\\
     h_{0,s}(x) & =s_{i}\left(\rho_{i,0}+\sum_{d=1}^{N_{D}}\rho_{i,d}D_{d}(x)\right)\;\;\;,\,i=N_{\theta}+s,\;s\in[1..N_{h}]
     
     \end{cases}
     
where :math:`s_{i=1..N_{\theta}+N_h}` is a scaled sigmoide, that is used because it is a bounded values function, and writes as follows:

.. math::
    :nowrap:
    
        \begin{eqnarray}
            
        s_i: \; \mathbb{R}& \mapsto \; &]l_i, u_i[\\
                x& \mapsto &l_{i} + \frac{u_{i}-l_{i}}{1 + e^{- x}}
                
       \end{eqnarray}


with :math:`l_i` and :math:`u_i` the bound constraint on the :math:`i^{th}` conceptual control such that :math:`\forall x \in \Omega`, for conceptual parameter 
:math:`l_i < \theta_i(x) < u_i, i\in[1..N_{\theta}]` and for conceptual states :math:`l_i < h_{0,s} < u_i,\; i=N_{\theta}+s, \; s\in [1..N_h]`.

.. note::

     This definition is practical since the sigmoid function is part of the mapping :math:`phi`, hence into the differentiated forward model, and enables to (i) bound the result of the descriptors to conceptual parameters mapping directly into the forward model, (ii) with a priori conceptual values, (iii) avoid handling complex variable change for bound constraining an optimization algorithm.
     
     
First Guess     
***********

The first guess :math:`\boldsymbol{\rho}^*` on the control vector :math:`\boldsymbol{\rho}` to optimize is defined accordingly, assuming a simple spatially uniform value, such that :math:`\forall x\in\Omega`: 

.. math::


     \rho_{k} =\left(s_{k}^{-1}\overline{\theta_{k}(x)},0,\;..\;0\right)	\;\;\;k\in[1..N_{\theta}] 
     
     \rho_{N_{\theta}+s} =\left(s_{k}^{-1}\overline{h_{0,s}(x)},0,\;..\;0\right)	\;\;\;s\in[1..N_{h}]
     
where :math:`s^{-1}_{i=1..N_{\theta}+N_h}` is a scaled inverse sigmoide that writes as follows:

.. math::
    :nowrap:
        
        \begin{eqnarray}

        s_i^{-1}: \; ]l_i, u_i[& \mapsto \; &\mathbb{R}\\
                     x& \mapsto &\ln\left(\frac{x - l_i}{u_i - x}\right)
                     
        \end{eqnarray}


with :math:`l_i` and :math:`u_i` the bound constraint on the :math:`i^{th}` conceptual control such that :math:`\forall x \in \Omega`, for conceptual parameter 
:math:`l_i < \theta_i(x) < u_i, i\in[1..N_{\theta}]` and for conceptual states :math:`l_i < h_{0,s} < u_i,\; i=N_{\theta}+s, \; s\in [1..N_h]`.


Multi-polynomial
----------------

This mapping is analoguous to the multi-linear mapping but with optimizable exponents applied to each physical descriptor.


Mapping formulation
*******************

The control vector :math:`\boldsymbol{\rho}` and the physical despcriptors :math:`\boldsymbol{\mathcal{D}}=\left(D_{d}(x)\right),\,d\in[1..N_{D}]` 
is mapped in order to impose spatially distributed parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h_0}`, such that :math:`\forall x\in\Omega`:  

.. math::

      \phi:\begin{cases}
      
      \theta_{k}(x) & =s_{k}\left(\rho_{k,0}+\sum_{d=1}^{N_{D}}\rho_{k,d}D_{d}^{\rho_{k,N_{D}+d}}(x)\right)\;\;\;,\,k\in[1..N_{\theta}]\\
      h_{0,s}(x) & =s_{i}\left(\rho_{i,0}+\sum_{d=1}^{N_{D}}\rho_{i,d}D_{d}^{\rho_{i,N_{D}+d}}(x)\right)\;\;\;,\,i=N_{\theta}+s,\;s\in[1..N_{h}]
      
      \end{cases}
     
The scaled sigmoide :math:`s_{\square}` is expressed in the :ref:`Multi-linear <math_num_documentation.mapping.multi_linear>` section.     
     
First Guess     
***********

The first guess :math:`\boldsymbol{\rho}^*` on the control vector :math:`\boldsymbol{\rho}` to optimize is defined accordingly, assuming a simple spatially uniform value, such that :math:`\forall x\in\Omega`: 

.. math::


     \rho_{k} =\left(s_{k}^{-1}\overline{\theta_{k}(x)},0,\;..\;0,\;1,\;..\;1\right)	\;\;\;k\in[1..N_{\theta}] 
     
     \rho_{N_{\theta}+s} =\left(s_{k}^{-1}\overline{h_{0,s}(x)},0,\;..\;0,\;1,\;..\;1\right)	\;\;\;s\in[1..N_{h}]
     
The scaled inverse sigmoide :math:`s_{\square}` is expressed in the :ref:`Multi-linear <math_num_documentation.mapping.multi_linear>` section.     


ANN
---

An artificial neural network (ANN) is used to map physical despcriptors :math:`\boldsymbol{\mathcal{D}}` onto conceptual parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h_0}` of the conceptual model.


Mapping formulation
*******************

The control vector :math:`\boldsymbol{\rho}` is mapped onto spatially distributed parameters :math:`\boldsymbol{\theta}` and 
initial states :math:`\boldsymbol{h}` using an (ANN) denoted :math:`\mathcal{N}`. It consists of a multilayer perceptron aiming to learn the physiographic descriptors :math:`\boldsymbol{\mathcal{D}}` to parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h}` mapping. In this case, :math:`\forall x\in\Omega`, the mapping :math:`\phi` writes:

.. math::

        \phi: [\boldsymbol{\theta}, \boldsymbol{h}_0](x) = \mathcal{N}\left(\boldsymbol{\mathcal{D}}(x), \boldsymbol{\rho} \right)
                 

where :math:`\boldsymbol{\rho} = [\boldsymbol{W}, \boldsymbol{b}]`, with :math:`\boldsymbol{W}` and :math:`\boldsymbol{b}` respectively the :math:`N_W` optimizable weights and :math:`N_b` biases of the neural network :math:`\mathcal{N}` composed of 
:math:`N_L` dense layers.

Note that an output layer consisting in a transformation based on the sigmoid function enables to impose bounds as previously such that for conceptual parameter  :math:`l_i < \theta_i(x) < u_i, i\in[1..N_{\theta}]` and for conceptual states :math:`l_i < h_{0,s} < u_i,\; i=N_{\theta}+s, \; s\in [1..N_h]`.

The following figure illustrates the architecture of the ANN with three hidden layers, followed by the ReLU activation function, 
and an output layer that uses the Sigmoid activation function in combination with a scaling function. In this particular case,
we have :math:`N_{\mathcal{D}} = 7` and :math:`N_{\theta} + N_{h} = 4`.

.. image:: ../_static/FCNN.png
    :width: 750
    :align: center


First guess
***********

In this case, the control vector :math:`\boldsymbol{\rho}`, representing the weights and biases of the ANN, is randomly initialized using one of the following methods:

- Zero initialization: :math:`\boldsymbol{\rho} = \mathbf{0}`
- Default uniform initialization: :math:`\boldsymbol{\rho} \sim \mathcal{U}\left(-\sqrt{\frac{1}{n_{in}}}, \sqrt{\frac{1}{n_{in}}}\right)`
- He uniform initialization: :math:`\boldsymbol{\rho} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)`
- Glorot uniform initialization: :math:`\boldsymbol{\rho} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)`
- Default normal initialization: :math:`\boldsymbol{\rho} \sim \mathcal{G}(0, 0.01)`
- He normal initialization: :math:`\boldsymbol{\rho} \sim \mathcal{G}(0, \sqrt{\frac{2}{n_{in}}})`
- Glorot normal initialization: :math:`\boldsymbol{\rho} \sim \mathcal{G}(0, \sqrt{\frac{2}{n_{in} + n_{out}}})`

with :math:`n_{in}` and :math:`n_{out}` denoting the number of neurons in the input and output layers, respectively; :math:`\mathcal{U}` and :math:`\mathcal{G}` denote the uniform and Gaussian distributions, respectively.


