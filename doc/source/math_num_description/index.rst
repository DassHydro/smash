.. _math_num_description:

======================
Math / Num Description
======================

.. warning::

    Section in development (redispatcher prose sur page de garde.. + mentioner d'entree hybrid NN, vda...)

`smash` is a computational software framework dedicated to **S**\patially distributed **M**\odelling and **AS**\similation for **H**\ydrology, enabling to tackle flexible spatially distributed hydrological modeling, signatures and sensitivity analysis, as well as high dimensional inverse problems using multi-source observations. This model is designed to simulate discharge hydrographs and hydrological states at any spatial location within a basin and reproduce the hydrological response of contrasted catchments, both for operational forecasting of floods :cite:p:`javelle_setting_2016` and low flows :cite:p:`Folton_2020`, by taking advantage of spatially distributed meteorological forcings, physiographic data and hydrometric observations.

`smash` is a modular platform that contains conceptual representations and numerical approximations of dominant hydrological processes while aiming to maintain a relative parsimony. It also contains several algorithms for signal analysis, model optimization and data assimilation over large datasets. It originally enables to use variational data assimilation and hybrid methods based on statistical/machine learning. All `smash` source files are written in Fortran and wrapped in Python using f90wrap :cite:p:`Kermode2020-f90wrap`. The adjoint code of the forward model is automatically generated using the differentiation tool Tapenade :cite:p:`hascoet2013tapenade` and some final tricks. The adjoint is used in the variational data assimilation algorithm presented in :cite:`jay2019potential`. `smash` enables to work at multiple spatio-temporal resolutions, with heterogeneous data in nature and sampling.

This documentation details the available forward model operators, signal and sensitivity analysis tools, and optimization algorithms.


Forward Model
*************

The forward spaially distributed hydrological modeling problem, several operators and predefined structures available in `smash` are described below.

.. toctree::
    :maxdepth: 1
    
    forward/forward_problem_statement
    forward/hydrological_operators
    forward/regionalization_operators

Signal analysis
***************

.. toctree::
    :maxdepth: 1
    
    signal_analysis/hydrograph_segmentation
    signal_analysis/hydrological_signatures
    signal_analysis/cost_functions

Inverse Algorithms
******************

.. toctree::
    :maxdepth: 1
    
    inverse/inverse_problem_statement
    inverse/algorithms

Data Preprocessing
******************

.. recup / preproc dem ; data sat ; NN for descriptors and other data preproc/filtering/selection...
