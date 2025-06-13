.. _user_guide:

==========
User Guide
==========

Data and Format Description
---------------------------
First, this section provides a description of the input data and their required format used in `smash`.

.. toctree::
    :maxdepth: 1

    data_and_format_description/cance
    data_and_format_description/france
    data_and_format_description/lez
    data_and_format_description/format_description


Quickstart
----------
This section provides an introductory walkthrough for new users of `smash`.
It includes beginner tutorials to help users get started quickly and become familiar with basic functionalities.

.. toctree::
    :maxdepth: 1

    quickstart/hydrological_mesh_construction
    quickstart/model_object_initialization
    quickstart/forward_run_classical_calibration

Classical Uses
--------------
This section highlights classical and well-established applications of `smash`.
It covers standard workflows, including calibration, validation techniques, and common hydrological analyses.

.. toctree::
    :maxdepth: 1

    classical_uses/split_sample_temporal_validation
    classical_uses/fully_distributed_calibration
    classical_uses/regionalization_spatial_validation
    classical_uses/large_domain_simulation
    classical_uses/forecasting_application
    classical_uses/hydrograph_segmentation
    classical_uses/hydrological_signatures
    classical_uses/rainfall_indices

In-depth
--------
This section dives into the advanced features and capabilities of `smash`, exploring more complex functionalities, and
providing detailed guidance for experienced users looking to maximize the software's potential.

.. toctree::
    :maxdepth: 1

    in_depth/retrieving_control_parameters
    in_depth/multicriteria_calibration
    in_depth/multisite_calibration
    in_depth/multiset_parameters_estimate
    in_depth/calibration_with_regularization_term
    in_depth/advanced_learnable_regionalization
    in_depth/hybrid_process_parameterization
    in_depth/bayesian_estimation

Post-processing and Interfacing `smash` with External Tools
-----------------------------------------------------------
This section focuses on post-processing in `smash` and how to use its outputs with external tools for extended functionality.

.. toctree::
    :maxdepth: 1

    post_processing_external_tools/results_visualization_over_large_sample
    post_processing_external_tools/sensitivity_analysis
    post_processing_external_tools/discharge_correction_lstm
