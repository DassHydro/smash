.. raw:: html

    <div align="center">
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DassHydro/smash/main/doc/source/_static/corporate_logo_dark.svg">
          <img src="https://raw.githubusercontent.com/DassHydro/smash/main/doc/source/_static/corporate_logo.svg" width="300" hspace="30">
        </picture>
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DassHydro/smash/main/doc/source/_static/logo_smash_dark.svg">
          <img src="https://raw.githubusercontent.com/DassHydro/smash/main/doc/source/_static/logo_smash.svg" width="300" hspace="30">
        </picture>
    </div>

.. image:: https://badge.fury.io/py/hydro-smash.svg
   :target: https://badge.fury.io/py/hydro-smash
   :alt: PyPI

**smash** is a Python library that provides a variety of user-friendly routines for hydrological modeling, including data preprocessing, high-dimensional optimization tools, sensitivity and signatures analysis—all interfaced with an efficient Fortran computational engine.

- **Documentation:** https://smash.recover.inrae.fr
- **Source code:** https://github.com/DassHydro/smash
- **Contributing:** https://smash.recover.inrae.fr/contributor_guide/index.html
- **Scientific references from latest version:** https://smash.recover.inrae.fr/bibliography/index.html
- **Bug reports:** https://github.com/DassHydro/smash/issues

It offers a range of advanced optimization techniques, including Variational Data Assimilation (VDA), Bayesian estimation, and Artificial Neural Network (ANN) approaches, using an adjoint model generated with the **Tapenade** automatic differentiation tool.

- **Tapenade** website: https://team.inria.fr/ecuador/en/tapenade/
- **Tapenade** article: https://doi.org/10.1145/2450153.2450158
- **Tapenade** source code: https://gitlab.inria.fr/tapenade/tapenade.git

Whether you are managing water resources or conducting research on hydrological models, **smash** can provide an easy-to-use yet powerful solution for your work. See the `getting started guide <https://smash.recover.inrae.fr/getting_started/index.html>`_ for installation instructions.

.. include:: CITATIONS.rst