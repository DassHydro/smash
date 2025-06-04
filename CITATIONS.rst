How to cite smash
=================

For **smash** software use, please cite:

    Colleoni, F., Huynh, N. N. T., Garambois, P.-A., Jay-Allemand, M., Organde, D., Renard, B., De Fournas, T., El Baz, A., Demargne, J., and Javelle, P. (2025). 
    SMASH v1.0: A Differentiable and Regionalizable High-Resolution Hydrological Modeling and Data Assimilation Framework. 
    EGUsphere, 2025, 1–36. 
    `<https://doi.org/10.5194/egusphere-2025-690>`_.

BibTeX entry:

.. code-block:: bibtex

    @article{Colleoni2025smash,
        author  = {Colleoni, François and Huynh, Ngo Nghi Truyen and Garambois, Pierre-André and Jay-Allemand, Maxime and Organde, Didier and Renard, Benjamin and De Fournas, Thomas and El Baz, Apolline and Demargne, Julie and Javelle, Pierre},
        title   = {SMASH v1.0: A Differentiable and Regionalizable High-Resolution Hydrological Modeling and Data Assimilation Framework},
        journal = {EGUsphere},
        volume  = {2025},
        year    = {2025},
        pages   = {1--36},
        doi     = {10.5194/egusphere-2025-690}
    }

.. TODO: update citation once accepted.

Please also cite the relevant references corresponding to the algorithms and methods used:

- Hybrid process-parameterization for water flux corrections:

    Huynh, N. N. T., Garambois, P.-A., Renard, B., Colleoni, F., Monnier, J., and Roux, H. (2025). 
    A Distributed Hybrid Physics-AI Framework for Learning Corrections of Internal Hydrological Fluxes and Enhancing High-Resolution Regionalized Flood Modeling. 
    EGUsphere, 2025, 1–34. 
    `<https://doi.org/10.5194/egusphere-2024-3665>`_.

- Hybrid regionalization, HDA-PR algorithm:

    Huynh, N. N. T., Garambois, P.-A., Colleoni, F., Renard, B., Roux, H., Demargne, J., Jay-Allemand, M., and Javelle, P. (2024). 
    Learning Regionalization Using Accurate Spatial Cost Gradients Within a Differentiable High-Resolution Hydrological Model: Application to the French Mediterranean Region. 
    Water Resources Research, 60, e2024WR037544. 
    `<https://doi.org/10.1029/2024WR037544>`_.

- Signatures, multi-criteria calibration, hydrograph segmentation algorithm:

    Huynh, N. N. T., Garambois, P.-A., Colleoni, F., and Javelle, P. (2023). 
    Signatures-and-sensitivity-based multi-criteria variational calibration for distributed hydrological modeling applied to Mediterranean floods. 
    Journal of Hydrology, 625, 129992. 
    `<https://doi.org/10.1016/j.jhydrol.2023.129992>`_.

- Fully distributed variational calibration:

    Jay-Allemand, M., Javelle, P., Gejadze, I., Arnaud, P., Malaterre, P.-O., Fine, J.-A., and Organde, D. (2020). 
    On the potential of variational calibration for a fully distributed hydrological model: application on a Mediterranean catchment. 
    Hydrology and Earth System Sciences, 24, 5519–5538. 
    `<https://doi.org/10.5194/hess-24-5519-2020>`_.

.. TODO: update ref flux correction once accepted and add ref neural ODE.

Related papers
==============

Additional **smash**-related publications:

    Garambois, P.A., Colleoni, F., Huynh, N. N. T., Akhtari, A., Nguyen, N. B., El Baz, A., Jay-Allemand, M., and Javelle, P. (2025). 
    Spatially Distributed Gradient-Based Calibration and Parametric Sensitivity of a Spatialized Hydrological Model Over 235 French Catchments. 
    HAL preprint, open science. 
    `<https://hal.science/hal-04989183>`_.
    
    Ettalbi, M., Garambois, P.A., Huynh, N. N. T., Arnaud, P., Ferreira, E., and Baghdadi, N. (2025). 
    Improving parameter regionalization learning for spatialized differentiable hydrological models by assimilation of satellite-based soil moisture data. 
    Journal of Hydrology, 660, 133300. 
    `<https://doi.org/10.1016/j.jhydrol.2025.133300>`_.

    Jay‐Allemand, M., Demargne, J., Garambois, P.‐A., Javelle, P., Gejadze, I., Colleoni, F., Organde, D., Arnaud, P., and Fouchier, C. (2024). 
    Spatially distributed calibration of ahydrological model with variational optimization constrained by physiographic maps for flash flood forecasting in France. 
    Proceedings of IAHS, 385, 281–290. 
    `<https://doi.org/10.5194/piahs-385-281-2024>`_.

    Evin, G., Le Lay, M., Fouchier, C., Penot, D., Colleoni, F., Mas, A., Garambois, P.-A., Laurantin, O. (2024).
    Evaluation of hydrological models on small mountainous catchments: impact of the meteorological forcings. 
    Hydrology and Earth System Sciences, 28, 261–281. 
    `<https://doi.org/10.5194/hess-28-261-2024>`__.

    Huynh, N. N. T., Garambois, P.‐A., Colleoni, F., Renard, B., and Roux, H. (2023). 
    Multi‐gauge hydrological variational data assimilation:Regionalization learning with spatial gradients using multilayer perceptron and Bayesian‐guided multivariate regression. 
    Colloque SHF 2023 - Prévision des crues et des inondations. 
    `<https://doi.org/10.48550/arXiv.2307.02497>`_.

.. TODO: update ref Garambois et al. 235bv once accepted.

Download smash references
=========================

:download:`smash.bib <javascript:downloadFile('https://raw.githubusercontent.com/DassHydro/smash/main/smash.bib', 'smash.bib')>`

.. raw:: html

   <script>
   function downloadFile(url, filename) {
     fetch(url)
       .then(response => response.blob())
       .then(blob => {
         const link = document.createElement('a');
         link.href = URL.createObjectURL(blob);
         link.download = filename;
         document.body.appendChild(link);
         link.click();
         document.body.removeChild(link);
       });
     return false;
   }
   </script>
