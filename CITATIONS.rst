How to cite smash
=================

For **smash** software use, please cite:

    Colleoni, F., Huynh, N. N. T., Garambois, P.-A., Jay-Allemand, M., Organde, D., Renard, B., De Fournas, T., El Baz, A., Demargne, J., and Javelle, P. (2025). 
    SMASH v1.0: A Differentiable and Regionalizable High-Resolution Hydrological Modeling and Data Assimilation Framework. 
    Geosci. Model Dev., 18, 2025, 7003–7034. 
    `<https://doi.org/10.5194/gmd-18-7003-2025>`_.

BibTeX entry:

.. code-block:: bibtex

    @article{Colleoni2025smash,
        author  = {Colleoni, François and Huynh, Ngo Nghi Truyen and Garambois, Pierre-André and Jay-Allemand, Maxime and Organde, Didier and Renard, Benjamin and De Fournas, Thomas and El Baz, Apolline and Demargne, Julie and Javelle, Pierre},
        title   = {SMASH v1.0: a differentiable and regionalizable high-resolution hydrological modeling and data assimilation framework},
        journal = {Geoscientific Model Development},
        volume  = {18},
        year    = {2025},
        number  = {19},
        pages   = {7003--7034},
        doi     = {10.5194/gmd-18-7003-2025}
    }

Please also cite the relevant references corresponding to the algorithms and methods used:

- Hybrid physics-AI framework for learning regionalization and refining internal water fluxes of algebraic or ordinary differential equations (ODEs)-based solvers:

    Huynh, N. N. T., Garambois, P.-A., Renard, B., Colleoni, F., Monnier, J., and Roux, H. (2025). 
    A distributed hybrid physics–AI framework for learning corrections of internal hydrological fluxes and enhancing high-resolution regionalized flood modeling. 
    Hydrol. Earth Syst. Sci., 29, 3589–3613. 
    `<https://doi.org/10.5194/hess-29-3589-2025>`_.

    Huynh, N. N. T., Garambois, P.-A., Colleoni, F., and Monnier, J. (2025). 
    Hybrid Physics-AI and Neural ODE Approaches for Spatially Distributed Hydrological Modeling. 
    EGUsphere, 2025, 1–24. 
    `<https://doi.org/10.5194/egusphere-2025-2797>`_.

- Hybrid Data Assimilation and Parameter Regionalization (HDA-PR) approach:

    Huynh, N. N. T., Garambois, P.-A., Colleoni, F., Renard, B., Roux, H., Demargne, J., Jay-Allemand, M., and Javelle, P. (2024). 
    Learning Regionalization Using Accurate Spatial Cost Gradients Within a Differentiable High-Resolution Hydrological Model: Application to the French Mediterranean Region. 
    Water Resour. Res., 60, e2024WR037544. 
    `<https://doi.org/10.1029/2024WR037544>`_.

- Signatures, multi-criteria calibration, hydrograph segmentation algorithm:

    Huynh, N. N. T., Garambois, P.-A., Colleoni, F., and Javelle, P. (2023). 
    Signatures-and-sensitivity-based multi-criteria variational calibration for distributed hydrological modeling applied to Mediterranean floods. 
    J. Hydrol., 625, 129992. 
    `<https://doi.org/10.1016/j.jhydrol.2023.129992>`_.

- Fully distributed variational calibration:

    Jay-Allemand, M., Javelle, P., Gejadze, I., Arnaud, P., Malaterre, P.-O., Fine, J.-A., and Organde, D. (2020). 
    On the potential of variational calibration for a fully distributed hydrological model: application on a Mediterranean catchment. 
    Hydrol. Earth Syst. Sci., 24, 5519–5538. 
    `<https://doi.org/10.5194/hess-24-5519-2020>`_.

Related papers
==============

Additional **smash**-related publications:

    Garambois, P.A., Colleoni, F., Huynh, N. N. T., Akhtari, A., Nguyen, N. B., El Baz, A., Jay-Allemand, M., and Javelle, P. (2025). 
    Spatially distributed gradient-based calibration and parametric sensitivity of a spatialized hydrological model over 235 French catchments. 
    J. Hydrol. : Reg. Stud., 60, 102485. 
    `<https://doi.org/10.1016/j.ejrh.2025.102485>`_.
    
    Ettalbi, M., Garambois, P.A., Huynh, N. N. T., Arnaud, P., Ferreira, E., and Baghdadi, N. (2025). 
    Improving parameter regionalization learning for spatialized differentiable hydrological models by assimilation of satellite-based soil moisture data. 
    J. Hydrol., 660, 133300. 
    `<https://doi.org/10.1016/j.jhydrol.2025.133300>`_.

    Jay‐Allemand, M., Demargne, J., Garambois, P.‐A., Javelle, P., Gejadze, I., Colleoni, F., Organde, D., Arnaud, P., and Fouchier, C. (2024). 
    Spatially distributed calibration of ahydrological model with variational optimization constrained by physiographic maps for flash flood forecasting in France. 
    Proceedings of IAHS, 385, 281–290. 
    `<https://doi.org/10.5194/piahs-385-281-2024>`_.

    Evin, G., Le Lay, M., Fouchier, C., Penot, D., Colleoni, F., Mas, A., Garambois, P.-A., Laurantin, O. (2024).
    Evaluation of hydrological models on small mountainous catchments: impact of the meteorological forcings. 
    Hydrol. Earth Syst. Sci., 28, 261–281. 
    `<https://doi.org/10.5194/hess-28-261-2024>`__.

    Huynh, N. N. T., Garambois, P.‐A., Colleoni, F., Renard, B., and Roux, H. (2023). 
    Multi‐gauge hydrological variational data assimilation:Regionalization learning with spatial gradients using multilayer perceptron and Bayesian‐guided multivariate regression. 
    Colloque SHF 2023 - Prévision des crues et des inondations. 
    `<https://doi.org/10.48550/arXiv.2307.02497>`_.

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
