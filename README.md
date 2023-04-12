## <img src="doc/source/_static/logo_smash.svg" width=180 align="center" alt=""/> - Spatially distributed Modelling and ASsimilation for Hydrology
[![Build Status](https://img.shields.io/badge/docs-public-brightgreen)](https://gitlab.irstea.fr/hydrology/smash)
    
## Compilation Instructions on Conda Environment

1.  Clone the SMASH repository from GitLab.
    ```bash
    git clone https://gitlab.irstea.fr/hydrology/smash.git
    ```
2.  Create the conda environment `(smash)`
    ```bash
    conda env create environment.yml
    ```
3.  Activate the conda environment `(smash)`
    ```bash
    conda activate smash
    ```
4.  Compile all programs, modules and libraries.
    ```bash
    (smash) make
    ```
5.  Check install
    ```bash
    python3
    ```
    ```python
    import smash
    ```
    
# Developer notes:

## Developer Environments

1.  The conda environment `(smash-dev)`
    ```bash
    conda env create environment-dev.yml
    ```
    
2. The docker environment

   A pre-filled in `Dockerfile` is available
   ```bash
   docker build --network=host -t smash-dev .
   ```
   ```bash
   docker run smash-dev
   ```
  
## Compile Adjoint and Tangent Linear Model

Make sure Java is installed (already done in docker and conda environment)

```bash
make tap
```
    
## Compile Code in Debug Mode
    
```bash
make debug
```

## Install `smash` Library in Editor Mode

The `smash` library is automatically installed in editor mode with the debug mode. Otherwise, one can switch to editor mode after compiling:

```bash
make library_edit
```
