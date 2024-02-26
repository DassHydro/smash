.. _math_num_documentation.regularization_function:

=======================
Regularization Function
=======================

The aim of this section is to present all the regularization functions that can be used as part of the regularization term in the inverse problem.

Denote :math:`\rho_k` the control vector associated to the :math:`k^{\text{th}}` parameter :math:`\boldsymbol{\theta}` or initial state :math:`\boldsymbol{h}`,
:math:`\rho_{0_k}` the prior value associated to :math:`\rho_k` and :math:`\phi` the mapping operator (see section :ref:`math_num_documentation.mapping`).

prior
-----

A reminder to prior value

.. math::

    j_{prior} = \sum_{k=1}^{N_{\theta} + N_h} \sum_{i=1}^N \left(\rho_k(i) - \rho_{0_k}(i)\right)^2

smoothing
---------

A spatial smoothing taking into account the prior value :math:`\rho_{0_k}`

.. math::
    :nowrap:

    \begin{eqnarray}

        &m_k& &=& &\phi(\rho_k) - \phi(\rho_{0_k})\\
        &j_{smoothing}& &=& &\sum_{k=1}^{N_{\theta} + N_h} \sum_{x=1}^{N_x} \sum_{y=1}^{N_y} 
        &\left(m_k(x - 1, y) - 2m_k(x, y) + m_k(x + 1, y)\right)^2 \\ 
        &&&&&&+ \left(m_k(x, y - 1) - 2m_k(x, y) + m_k(x, y + 1)\right)^2

    \end{eqnarray}

with the following boundary conditions

.. math::
    :nowrap:

    \begin{eqnarray}

        &&m_k(N_x + 1, y)& =& &m_k(N_x, y)\\
        &&m_k(0, y)& =& &m_k(1, y)\\
        &&m_k(x, N_y + 1)& =& &m_k(x, N_y)\\
        &&m_k(x, 0)& =& &m_k(x, 1)\\
        &&m_k(N_x + 1, N_y + 1)& =& &m_k(N_x, N_y)\\
        &&m_k(0, 0)& =& &m_k(1, 1)\\
        &&m_k(N_x + 1, 0)& =& &m_k(N_x, 1)\\
        &&m_k(0, N_y + 1)& =& &m_k(1, N_y)

    \end{eqnarray}

hard-smoothing
--------------

A spatial smoothing without taking into account the prior value :math:`\rho_{0_k}`

.. math::
    :nowrap:

    \begin{eqnarray}

        &m_k& &=& &\phi(\rho_k)\\
        &j_{hard-smoothing}& &=& &j_{smoothing}

    \end{eqnarray}
