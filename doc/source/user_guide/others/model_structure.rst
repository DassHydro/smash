.. _user_guide.model_structure:

===============
Model structure
===============

In this section all the hydrological model structures that can be used will be presented.

There are 3 different structures available:

- "gr-a"
    4 parameters and 3 states structure derived from the GR model.
    
- "gr-b"
    4 parameters and 4 states structure derived from the GR model.
    
- "gr-c"
    5 parameters and 5 states structure derived from the GR model.
    

.. note::
    see the Math / Num description section: (**TODO** :ref:`math_num_description`) for more information about GR model.
    

Model structure description
---------------------------

.. _user_guide.model_structure.gr_a:

gr-a
''''

.. image:: ../../_static/diagram_gr-a.png
    :width: 400
    :align: center
    
Parameters
**********

- ``cp``: the maximum capacity of the production storage :math:`(mm)`,
- ``cft``: the maximum capacity of the transfer storage :math:`(mm)`,
- ``exc``: the non-conservative exchange parameter :math:`(mm/dt)`,
- ``lr``: the linear routing parameter :math:`(min)`.

States
******

- ``hp``: the relative state of the production storage :math:`(-)`,
- ``hft``: the relative state of the transfer storage :math:`(-)`,
- ``hr``: the absolute state of the routing storage :math:`(mm)`.

Operating
*********

- neutralization of :math:`P` by :math:`E` to determine a net rainfall :math:`P_n` and a net evapotranspiration :math:`E_n`,
- filling (resp. emptying) the production storage by :math:`P_s` (resp. :math:`E_s`),
- splitting :math:`P_r` into two branches, 90% filling the transfer storage and 10% into the direct branch,
- application of the non-conservative flux :math:`F` (which can be either positive or negative) in both branches,
- summing :math:`Q_r`, the outgoing flux of the transfer storage and :math:`Q_d`, the outgoing flux of the direct branch giving the cell flux :math:`Q_t`,
- filling the routing storage by the upstream flux :math:`Q_{up}` and the cell flux :math:`Q_t`,
- calculation of the final routed flow :math:`Q` at the output of the routing storage.

gr-b
''''

.. image:: ../../_static/diagram_gr-b.png
    :width: 400
    :align: center
    
Parameters
**********

- ``cp``: the maximum capacity of the production storage :math:`(mm)`,
- ``cft``: the maximum capacity of the transfer storage :math:`(mm)`,
- ``exc``: the non-conservative exchange parameter :math:`(mm/dt)`,
- ``lr``: the linear routing parameter :math:`(min)`.

States
******

- ``hi``: the relative state of the interception storage :math:`(-)`,
- ``hp``: the relative state of the production storage :math:`(-)`,
- ``hft``: the relative state of the transfer storage :math:`(-)`,
- ``hr``: the absolute state of the routing storage :math:`(mm)`.

Operating
*********

- neutralization of :math:`P` by :math:`E` to determine a net rainfall :math:`P_n` and a net evapotranspiration :math:`E_n` using an interception storage,

.. note::
    In case of a daily time step simulation, the interception storage is disabled and the neutralization of :math:`P` by :math:`E` is similar to :ref:`user_guide.model_structure.gr_a`.
    Otherwise (at sub-daily time step), the maximum capacity :math:`c_i` is adjusted to match fluxes between the simulation at daily time and sub-daily time step.

- filling (resp. emptying) the production storage by :math:`P_s` (resp. :math:`E_s`),
- splitting :math:`P_r` into two branches, 90% filling the transfer storage and 10% into the direct branch,
- application of the non-conservative flux :math:`F` (which can be either positive or negative) in both branches,
- summing :math:`Q_r`, the outgoing flux of the transfer storage and :math:`Q_d`, the outgoing flux of the direct branch giving the cell flux :math:`Q_t`,
- filling the routing storage by the upstream flux :math:`Q_{up}` and the cell flux :math:`Q_t`,
- calculation of the final routed flow :math:`Q` at the output of the routing storage.

gr-c
''''

.. image:: ../../_static/diagram_gr-c.png
    :width: 425
    :align: center
    
Parameters
**********

- ``cp``: the maximum capacity of the production storage :math:`(mm)`,
- ``cft``: the maximum capacity of the first transfer storage :math:`(mm)`,
- ``cst``: the maximum capacity of the second transfer storage :math:`(mm)`,
- ``exc``: the non-conservative exchange parameter :math:`(mm/dt)`,
- ``lr``: the linear routing parameter :math:`(min)`.

States
******

- ``hi``: the relative state of the interception storage :math:`(-)`,
- ``hp``: the relative state of the production storage :math:`(-)`,
- ``hft``: the relative state of the transfer storage :math:`(-)`,
- ``hst``: the relative state of the transfer storage :math:`(-)`,
- ``hr``: the absolute state of the routing storage :math:`(mm)`.

Operating
*********

- neutralization of :math:`P` by :math:`E` to determine a net rainfall :math:`P_n` and a net evapotranspiration :math:`E_n` using an interception storage,

.. note::
    In case of a daily time step simulation, the interception storage is disabled and the neutralization of :math:`P` by :math:`E` is similar to :ref:`user_guide.model_structure.gr_a`.
    Otherwise (at sub-daily time step), the maximum capacity :math:`c_i` is adjusted to match fluxes between the simulation at daily time and sub-daily time step.

- filling (resp. emptying) the production storage by :math:`P_s` (resp. :math:`E_s`),
- splitting :math:`P_r` into three branches, 54% filling the first transfer storage, 36% filling the second transfer storage and 10% into the direct branch,
- application of the non-conservative flux :math:`F` (which can be either positive or negative) in the first transfer and direct branches,
- summing :math:`Q_r`, the outgoing flux of the first transfer storage, :math:`Q_l`, the outgoing flux of the second transfer storage and :math:`Q_d`, the outgoing flux of the direct branch giving the cell flux :math:`Q_t`,
- filling the routing storage by the upstream flux :math:`Q_{up}` and the cell flux :math:`Q_t`,
- calculation of the final routed flow :math:`Q` at the output of the routing storage.
