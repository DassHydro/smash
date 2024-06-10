:notoc:

.. module:: smash

*******************
smash documentation
*******************

**Date**: |today| **Version**: |version|

`smash` is a computational software framework dedicated to **S**\patially distributed **M**\odelling and
**AS**\similation for **H**\ydrology, enabling to tackle spatially distributed differentiable hydrological
modeling, with learnable parameterization-regionalization. This platform enables to combine vertical and
lateral flow operators, either process-based conceptual or hydrid  with neural networks, and perform high 
dimensional non linear optimization from multi-source data. It is designed to simulate discharge hydrographs 
and hydrological states at any spatial location within a basin and reproduce the hydrological response of
contrasted catchments, both for operational forecasting of floods and low flows, by taking advantage of
spatially distributed meteorological forcings, physiographic data and hydrometric observations.

.. grid:: 2

    .. grid-item-card::
        :img-top: _static/index_getting_started.svg
        
        Getting Started
        ^^^^^^^^^^^^^^^

        New to *smash*? Check out the getting started guides.

        +++

        .. button-ref:: getting_started
            :expand:
            :color: muted
            :click-parent:

            To the getting started
            
    .. grid-item-card::
        :img-top: _static/index_user_guide.svg

        User Guide
        ^^^^^^^^^^

        The user guide provides in depth-information of the *smash* library.

        +++

        .. button-ref:: user_guide
            :expand:
            :color: muted
            :click-parent:

            To the user guide
            
    .. grid-item-card::
        :img-top: _static/index_api.svg

        API Reference
        ^^^^^^^^^^^^^

        The reference guide contains a detailed description of
        the *smash* API.

        +++

        .. button-ref:: api_reference
            :expand:
            :color: muted
            :click-parent:

            To the api reference
            
    .. grid-item-card::
        :img-top: _static/index_math_num.svg

        Math / Num Documentation
        ^^^^^^^^^^^^^^^^^^^^^^^^

        This documentation contains a detailed description of the *smash* model operators and associated
        mathematical and numerical tools.

        +++

        .. button-ref:: math_num_documentation
            :expand:
            :color: muted 
            :click-parent:

            To the math / num doc

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   
   getting_started/index
   user_guide/index
   api_reference/index
   math_num_documentation/index
   release/index
   contributor_guide/index
   license/index
   bibliography/index
