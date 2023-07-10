from __future__ import annotations

from smash.io.mesh.mesh import read_mesh
from smash.io.setup.setup import read_setup

import os


__all__ = ["load_dataset"]

DATASET_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_NAME = ["flwdir", "cance", "lez", "france"]


def load_dataset(name: str):
    """
    Load dataset.

    A function allowing user to load different kinds of data or pre-filled files for a first use of the `smash` package.

    .. hint::
        See the :ref:`user_guide` cases section for more.

    Parameters
    ----------
    name : str
        The dataset name. Should be one of

        - 'flwdir' : The absolute path to a 1km² France flow directions in `smash` convention.
        - 'cance' : Setup and mesh dictionaries used to initialize the Model object on the Cance catchment at **hourly** timestep.
        - 'lez' : Setup and mesh dictionaries used to initialize the Model object on the Lez catchment at **daily** timestep.
        - 'france' : Setup and mesh dictionaries used to initialize the Model object on the France at **hourly** timestep.

    Returns
    -------
    dataset : str or tuple
        Depending on the dataset choosen

        - 'flwdir' : Returns a file path.
        - 'cance' : Returns a tuple of dictionaries (setup and mesh).
        - 'lez' : Returns a tuple of dictionaries (setup and mesh).
        - 'france' : Returns a tuple of dictionaries (setup and mesh).

    Examples
    --------

    Load ``flwdir`` dataset. (the path is updated for each user).

    >>> flwdir = smash.factory.load_dataset("flwdir")
    >>> flwdir
    '/home/fcolleoni/anaconda3/envs/smash-dev/lib/python3.8/site-packages/smash/dataset/France_flwdir.tif'

    Load ``cance`` dataset as a tuple of dictionaries.

    >>> cance = smash.factory.load_dataset("cance")
    >>> cance
    ({'structure': 'gr-a-lr', 'dt': 3600, ...}, {'nac': 383, 'ncol': 28, ...})

    Or each dictionary in a different variable.

    >>> setup, mesh = smash.factory.load_dataset("cance")
    >>> setup
    {'structure': 'gr-a-lr', 'dt': 3600, ...}
    >>> mesh
    {'nac': 383, 'ncol': 28, ...}

    Load ``lez`` dataset as a tuple of dictionaries.

    >>> lez = smash.factory.load_dataset("lez")
    >>> lez
    ({'structure': 'gr-a-lr', 'dt': 86400, ...}, {'nac': 172, 'ncol': 14, ...})

    Or each dictionary in a different variable.

    >>> setup, mesh = smash.factory.load_dataset("lez")
    >>> setup
    {'structure': 'gr-a-lr', 'dt': 86400, ...}
    >>> mesh
    {'nac': 172, 'ncol': 14, ...}

    Load ``france`` dataset as a tuple of dictionaries.

    >>> france = smash.factory.load_dataset("france")
    >>> france
    ({'structure': 'gr-a-lr', 'dt': 3600, ...}, {'nac': 906044, 'ncol': 1150, ...})

    Or each dictionary in a different variable.

    >>> setup, mesh = smash.factory.load_dataset("france")
    >>> setup
    {'structure': 'gr-a-lr', 'dt': 3600, ...}
    >>> mesh
    {'nac': 906044, 'ncol': 1150, ...}
    """

    if name.lower() == "flwdir":
        return os.path.join(DATASET_PATH, "France_flwdir.tif")

    elif name.lower() in ["cance", "lez", "france"]:
        cptl_name = name.capitalize()

        setup = read_setup(
            os.path.join(DATASET_PATH, cptl_name, f"setup_{cptl_name}.yaml")
        )
        mesh = read_mesh(
            os.path.join(DATASET_PATH, cptl_name, f"mesh_{cptl_name}.hdf5")
        )

        setup.update(
            {
                "qobs_directory": os.path.join(DATASET_PATH, cptl_name, "qobs"),
                "prcp_directory": os.path.join(DATASET_PATH, cptl_name, "prcp"),
                "pet_directory": os.path.join(DATASET_PATH, cptl_name, "pet"),
                "descriptor_directory": os.path.join(
                    DATASET_PATH, cptl_name, "descriptor"
                ),
            }
        )

        return setup, mesh

    else:
        raise ValueError(f"Unknown dataset '{name}'. Choices: {DATASET_NAME}")
