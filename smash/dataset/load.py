from __future__ import annotations

import smash
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
        - 'path/to/dataset' : Path to an external and complete dataset.

    Returns
    -------
    dataset : str or tuple
        Depending on the dataset choosen

        - 'flwdir' : Returns a file path.
        - 'cance' : Returns a tuple of dictionaries (setup and mesh).
        - 'lez' : Returns a tuple of dictionaries (setup and mesh).
        - 'france' : Returns a tuple of dictionaries (setup and mesh).
        - 'path/to/dataset' : Returns a tuple of dictionaries (setup and mesh).

    Examples
    --------

    Load ``flwdir`` dataset. (the path is updated for each user).

    >>> flwdir = smash.load_dataset("flwdir")
    >>> flwdir
    '/home/francois/anaconda3/envs/smash-dev/lib/python3.8/site-packages/smash/dataset/France_flwdir.tif'

    Load ``cance`` dataset as a tuple of dictionaries.

    >>> cance = smash.load_dataset("cance")
    >>> cance
    ({'structure': 'gr-a', 'dt': 3600, ...}, {'dx': 1000.0, 'nac': 383, ...})

    Or each dictionary in a different variable.

    >>> setup, mesh = smash.load_dataset("cance")
    >>> setup
    {'structure': 'gr-a', 'dt': 3600, ...}
    >>> mesh
    {'dx': 1000.0, 'nac': 383, ...}

    Load ``lez`` dataset as a tuple of dictionaries.

    >>> lez = smash.load_dataset("lez")
    >>> lez
    ({'structure': 'gr-a', 'dt': 86400, ...}, {'dx': 1000.0, 'nac': 172, ...})

    Or each dictionary in a different variable.

    >>> setup, mesh = smash.load_dataset("lez")
    >>> setup
    {'structure': 'gr-a', 'dt': 86400, ...}
    >>> mesh
    {'dx': 1000.0, 'nac': 172, ...}

    Load ``france`` dataset as a tuple of dictionaries.

    >>> france = smash.load_dataset("france")
    >>> france
    ({'structure': 'gr-a', 'dt': 3600, ...}, {'dx': 1000.0, 'nac': 906044, ...})

    Or each dictionary in a different variable.

    >>> setup, mesh = smash.load_dataset("france")
    >>> setup
    {'structure': 'gr-a', 'dt': 3600, ...}
    >>> mesh
    {'dx': 1000.0, 'nac': 906044, ...}

    Load ``path/to/dataset`` as a tuple of dictionaries.

    >>> dataset = smash.load_dataset("path/to/dataset")
    >>> dataset
    ({'structure': 'gr-a', 'dt': 3600, ...}, {'dx': 1000.0, 'nac': 383, ...})

    Or each dictionary in a different variable.

    >>> setup, mesh = smash.load_dataset("path/to/dataset")
    >>> setup
    {'structure': 'gr-a', 'dt': 3600, ...}
    >>> mesh
    {'dx': 1000.0, 'nac': 383, ...}
    """

    if name.lower() == "flwdir":
        return os.path.join(DATASET_PATH, "France_flwdir.tif")

    elif name.lower() in ["cance", "lez", "france"]:
        cptl_name = name.capitalize()

        setup = smash.read_setup(
            os.path.join(DATASET_PATH, cptl_name, f"setup_{cptl_name}.yaml")
        )
        mesh = smash.read_mesh(
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

    # load an external dataset
    elif os.path.exists(name):
        local_dataset_path = os.path.dirname(os.path.realpath(name))
        local_dataset_dir_name = os.path.basename(os.path.realpath(name))
        setup_file = "setup_" + local_dataset_dir_name + ".yaml"
        mesh_file = "mesh_" + local_dataset_dir_name + ".hdf5"

        if not os.path.exists(
            os.path.join(local_dataset_path, local_dataset_dir_name, setup_file)
        ):
            raise ValueError(f"Missing setup file '{setup_file}'")

        if not os.path.exists(
            os.path.join(local_dataset_path, local_dataset_dir_name, mesh_file)
        ):
            raise ValueError(f"Missing mesh file '{mesh_file}'")

        setup = smash.read_setup(
            os.path.join(local_dataset_path, local_dataset_dir_name, setup_file)
        )
        mesh = smash.read_mesh(
            os.path.join(local_dataset_path, local_dataset_dir_name, mesh_file)
        )

        setup.update(
            {
                "qobs_directory": os.path.join(
                    local_dataset_path, local_dataset_dir_name, "qobs"
                ),
                "prcp_directory": os.path.join(
                    local_dataset_path, local_dataset_dir_name, "prcp"
                ),
                "pet_directory": os.path.join(
                    local_dataset_path, local_dataset_dir_name, "pet"
                ),
                "descriptor_directory": os.path.join(
                    local_dataset_path, local_dataset_dir_name, "descriptor"
                ),
            }
        )

        return setup, mesh

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Choices: {DATASET_NAME}. Or non existing external dataset"
        )
