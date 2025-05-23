<h1 align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DassHydro/smash/main/doc/source/_static/corporate_logo_dark.svg">
  <img src="https://raw.githubusercontent.com/DassHydro/smash/main/doc/source/_static/corporate_logo.svg" width="300", hspace=30>
</picture>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DassHydro/smash/main/doc/source/_static/logo_smash_dark.svg">
  <img src="https://raw.githubusercontent.com/DassHydro/smash/main/doc/source/_static/logo_smash.svg" width="300", hspace=30>
</picture>
</h1><br>

[![PyPI](https://badge.fury.io/py/hydro-smash.svg)](https://badge.fury.io/py/hydro-smash)

**smash** is a Python library that provides a variety of user-friendly routines for hydrological modeling, including data preprocessing, high-dimensional optimization tools, sensitivity and signatures analysis—all interfaced with an efficient Fortran computational engine.

- **Documentation:** https://smash.recover.inrae.fr
- **Source code:** https://github.com/DassHydro/smash
- **Contributing:** https://smash.recover.inrae.fr/contributor_guide/index.html
- **Citations and related papers:** https://smash.recover.inrae.fr/citations/index.html
- **Scientific references:** https://smash.recover.inrae.fr/bibliography/index.html
- **Bug reports:** https://github.com/DassHydro/smash/issues

It offers a range of advanced optimization techniques, including Variational Data Assimilation (VDA), Bayesian estimation, and Artificial Neural Network (ANN) approaches, using an adjoint model generated with the **Tapenade** automatic differentiation tool.

- **Tapenade** website: https://team.inria.fr/ecuador/en/tapenade
- **Tapenade** article: https://doi.org/10.1145/2450153.2450158
- **Tapenade** source code: https://gitlab.inria.fr/tapenade/tapenade.git

Whether you are managing water resources or conducting research on hydrological models, **smash** can provide an easy-to-use yet powerful solution for your work. See the [getting started](https://smash.recover.inrae.fr/getting_started/index.html) guide for installation instructions.

## How to cite smash

For **smash** software use, please cite:

> Colleoni, F., Huynh, N. N. T., Garambois, P.-A., Jay-Allemand, M., Organde, D., Renard, B., De Fournas, T., El Baz, A., Demargne, J., and Javelle, P. (2025). 
> SMASH v1.0: A Differentiable and Regionalizable High-Resolution Hydrological Modeling and Data Assimilation Framework. 
> EGUsphere, 2025, 1–36. 
> https://doi.org/10.5194/egusphere-2025-690.

BibTeX entry:

```bibtex
@article{Colleoni2025smash,
    author  = {Colleoni, François and Huynh, Ngo Nghi Truyen and Garambois, Pierre-André and Jay-Allemand, Maxime and Organde, Didier and Renard, Benjamin and De Fournas, Thomas and El Baz, Apolline and Demargne, Julie and Javelle, Pierre},
    title   = {SMASH v1.0: A Differentiable and Regionalizable High-Resolution Hydrological Modeling and Data Assimilation Framework},
    journal = {EGUsphere},
    volume  = {2025},
    year    = {2025},
    pages   = {1--36},
    doi     = {10.5194/egusphere-2025-690}
}
```

Please also cite the relevant references corresponding to the algorithms and methods used:

- Hybrid process-parameterization for water flux corrections:

    > Huynh, N. N. T., Garambois, P.-A., Renard, B., Colleoni, F., Monnier, J., and Roux, H. (2025). 
    > A Distributed Hybrid Physics-AI Framework for Learning Corrections of Internal Hydrological Fluxes and Enhancing High-Resolution Regionalized Flood Modeling. 
    > EGUsphere, 2025, 1–34. 
    > https://doi.org/10.5194/egusphere-2024-3665.

- Hybrid regionalization, HDA-PR algorithm:

    > Huynh, N. N. T., Garambois, P.-A., Colleoni, F., Renard, B., Roux, H., Demargne, J., Jay-Allemand, M., and Javelle, P. (2024). 
    > Learning Regionalization Using Accurate Spatial Cost Gradients Within a Differentiable High-Resolution Hydrological Model: Application to the French Mediterranean Region. 
    > Water Resources Research, 60, e2024WR037544. 
    > https://doi.org/10.1029/2024WR037544.

- Signatures, multi-criteria calibration, hydrograph segmentation algorithm:

    > Huynh, N. N. T., Garambois, P.-A., Colleoni, F., and Javelle, P. (2023). 
    > Signatures-and-sensitivity-based multi-criteria variational calibration for distributed hydrological modeling applied to Mediterranean floods. 
    > Journal of Hydrology, 625, 129992. 
    > https://doi.org/10.1016/j.jhydrol.2023.129992.

- Fully distributed variational calibration:

    > Jay-Allemand, M., Javelle, P., Gejadze, I., Arnaud, P., Malaterre, P.-O., Fine, J.-A., and Organde, D. (2020). 
    > On the potential of variational calibration for a fully distributed hydrological model: application on a Mediterranean catchment. 
    > Hydrology and Earth System Sciences, 24, 5519–5538. 
    > https://doi.org/10.5194/hess-24-5519-2020.
