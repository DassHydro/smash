.. _user_guide:

==========
User Guide
==========

Data and Format description
---------------------------
First, this section provides descriptions of the demo data that can be downloaded and will be used for the following tutorials.

.. toctree::
    :maxdepth: 1

    data_and_format_description/cance
    data_and_format_description/france
    data_and_format_description/lez
    data_and_format_description/working_with_your_own_data


Quickstart
----------
This section provides an introductory walkthrough for new users of `smash`.
It includes beginner tutorials to help users get started quickly and become familiar with basic functionalities.

.. toctree::
    :maxdepth: 1

    quickstart/input_data_model_initialization
    quickstart/simulation_model_response
    quickstart/calibration_visualization

Classical Uses
--------------
This section highlights classical and well-established applications of `smash`.
It covers standard workflows, including calibration, validation techniques, and common hydrological analyses.

.. toctree::
    :maxdepth: 1

    classical_uses/classical_calibration_io
    classical_uses/split_sample_temporal_validation
    classical_uses/regionalization_spatial_validation
    classical_uses/large_domain_simulation
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
    in_depth/hybrid_model_strutures
    in_depth/bayesian_estimation_approach

Post-processing and enterfacing `smash` with external tools
-----------------------------------------------------------
This section focuses on how to integrate `smash` outputs with external tools for extended functionality.

.. toctree::
    :maxdepth: 1

    post_processing_external_tools/reading_large_sample
    post_processing_external_tools/results_visualization
    post_processing_external_tools/optimize_parameters_analysis
    post_processing_external_tools/sensitivity_analysis
    post_processing_external_tools/discharge_correction_lstm
