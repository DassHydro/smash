from __future__ import annotations

import os

from smash._constant import DATASET_NAME
from smash.io.mesh import read_mesh
from smash.io.setup import read_setup

__all__ = ["load_dataset"]

DATASET_PATH = os.path.dirname(os.path.realpath(__file__))


# TODO: Maybe remove the dataset from the GitHub repository and fetch the data anywhere else (cloud, zenodo,
# etc)
# It requires internet connexion
def load_dataset(name: str):
    """
    Load dataset.

    A function allowing user to load different kinds of data or pre-filled files for a first use of the
    `smash` package.

    Parameters
    ----------
    name : `str`
        The dataset name. Should be one of

        - ``'flwdir'`` : The absolute path to a 1km² France flow directions in `smash` convention.
        - ``'cance'`` : Setup and mesh dictionaries used to initialize the Model object on the Cance catchment
          at ``hourly`` time step.
        - ``'lez'`` : Setup and mesh dictionaries used to initialize the Model object on the Lez catchment at
          ``daily`` time step.
        - ``'france'`` : Setup and mesh dictionaries used to initialize the Model object on the France at
          ``hourly`` time step.

    Returns
    -------
    dataset : `str` or `tuple[dict, dict]`
        Depending on the dataset choosen

        - ``'flwdir'`` : Returns a file path.
        - ``'cance'`` : Returns a tuple of dictionaries (setup and mesh).
        - ``'lez'`` : Returns a tuple of dictionaries (setup and mesh).
        - ``'france'`` : Returns a tuple of dictionaries (setup and mesh).

    Examples
    --------
    >>> from smash.factory import load_dataset

    Load ``'flwdir'`` dataset (the path is updated for each user):

    >>> flwdir = load_dataset("flwdir")
    >>> flwdir
    '/home/fcolleoni/Documents/git/smash/smash/factory/dataset/France_flwdir.tif'

    Load ``'cance'`` dataset as a tuple of dictionaries:

    >>> cance = load_dataset("cance")
    >>> cance
    ({'hydrological_module': 'gr4', 'routing_module': 'lr', ...}, {'nac': 383, 'ncol': 28, ...})

    Or each dictionary in a different variable:

    >>> setup, mesh = load_dataset("cance")
    >>> setup
    {'hydrological_module': 'gr4', 'routing_module': 'lr', ...}
    >>> mesh
    {'nac': 383, 'ncol': 28, ...}

    Load ``'lez'`` dataset as a tuple of dictionaries:

    >>> lez = load_dataset("lez")
    >>> lez
    ({'hydrological_module': 'gr4', 'routing_module': 'lr', ...}, {'nac': 172, 'ncol': 14, ...})

    Or each dictionary in a different variable:

    >>> setup, mesh = load_dataset("lez")
    >>> setup
    {'hydrological_module': 'gr4', 'routing_module': 'lr', ...}
    >>> mesh
    {'nac': 172, 'ncol': 14, ...}

    Load ``'france'`` dataset as a tuple of dictionaries:

    >>> france = load_dataset("france")
    >>> france
    ({'hydrological_module': 'gr4', 'routing_module': 'lr', ...}, {'nac': 906044, 'ncol': 1150, ...})

    Or each dictionary in a different variable:

    >>> setup, mesh = load_dataset("france")
    >>> setup
    {'hydrological_module': 'gr4', 'routing_module': 'lr', ...}
    >>> mesh
    {'nac': 906044, 'ncol': 1150, ...}
    """

    if name.lower() == "flwdir":
        return os.path.join(DATASET_PATH, "France_flwdir.tif")

    elif name.lower() in ["cance", "lez", "france"]:
        cptl_name = name.capitalize()

        setup = read_setup(os.path.join(DATASET_PATH, cptl_name, f"setup_{cptl_name}.yaml"))
        mesh = read_mesh(os.path.join(DATASET_PATH, cptl_name, f"mesh_{cptl_name}.hdf5"))

        setup.update(
            {
                "qobs_directory": os.path.join(DATASET_PATH, cptl_name, "qobs"),
                "prcp_directory": os.path.join(DATASET_PATH, cptl_name, "prcp"),
                "pet_directory": os.path.join(DATASET_PATH, cptl_name, "pet"),
                "descriptor_directory": os.path.join(DATASET_PATH, cptl_name, "descriptor"),
            }
        )

        return setup, mesh

    else:
        raise ValueError(f"Unknown dataset '{name}'. Choices: {DATASET_NAME}")
