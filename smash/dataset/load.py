from __future__ import annotations

import smash
import os


__all__ = ["load_dataset"]

DATASET_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_NAME = ["flwdir", "cance", "france"]


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
        - 'cance' : Setup and mesh dictionaries used to initialize the Model object on the Cance catchment.
        - 'france' : Setup and mesh dictionaries used to initialize the Model object on the France.

    Returns
    -------
    dataset : str or tuple
        Depending on the dataset choosen

        - 'flwdir' : Returns a file path.
        - 'cance' : Returns a tuple of dictionaries (setup and mesh).
        - 'france' : Returns a tuple of dictionaries (setup and mesh).

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
    """

    if name == "flwdir":
        return os.path.join(DATASET_PATH, "France_flwdir.tif")

    elif name == "cance":
        setup = smash.read_setup(os.path.join(DATASET_PATH, "Cance/setup_Cance.yaml"))
        mesh = smash.read_mesh(os.path.join(DATASET_PATH, "Cance/mesh_Cance.hdf5"))

        setup.update(
            {
                "qobs_directory": os.path.join(DATASET_PATH, "Cance/qobs"),
                "prcp_directory": os.path.join(DATASET_PATH, "Cance/prcp"),
                "pet_directory": os.path.join(DATASET_PATH, "Cance/pet"),
                "descriptor_directory": os.path.join(DATASET_PATH, "Cance/descriptor"),
            }
        )

        return setup, mesh

    elif name == "france":
        setup = smash.read_setup(os.path.join(DATASET_PATH, "France/setup_France.yaml"))
        mesh = smash.read_mesh(os.path.join(DATASET_PATH, "France/mesh_France.hdf5"))

        setup.update(
            {
                "prcp_directory": os.path.join(DATASET_PATH, "France/prcp"),
                "pet_directory": os.path.join(DATASET_PATH, "France/pet"),
            }
        )

        return setup, mesh

    #load an external dataset
    elif (os.path.exists(name)):
        
        local_dataset_path = os.path.dirname(os.path.realpath(name))
        local_dataset_dir_name = os.path.basename(os.path.realpath(name))
        config_file="setup_"+local_dataset_dir_name+".yaml"
        mesh_file="mesh_"+local_dataset_dir_name+".hdf5"
        
        if ( os.path.exists(os.path.join(local_dataset_path, local_dataset_dir_name, config_file)) == False ) :
            raise ValueError(f"Bad dataset '{name}'. Missing file '{config_file}'")
        
        if ( os.path.exists(os.path.join(local_dataset_path, local_dataset_dir_name, mesh_file)) == False ) :
            raise valueerror(f"bad dataset '{name}'. missing file '{mesh_file}'")
        
        setup = smash.read_setup(os.path.join(local_dataset_path, local_dataset_dir_name, config_file))
        mesh = smash.read_mesh(os.path.join(local_dataset_path,  local_dataset_dir_name, mesh_file))
        
        setup.update(
            {
                "qobs_directory": os.path.join(local_dataset_path,local_dataset_dir_name, "qobs"),
                "prcp_directory": os.path.join(local_dataset_path,local_dataset_dir_name, "prcp"),
                "pet_directory": os.path.join(local_dataset_path, local_dataset_dir_name, "pet"),
                "descriptor_directory": os.path.join(local_dataset_path, local_dataset_dir_name, "descriptor"),
            }
        )
        
        return setup, mesh

    else:
        raise ValueError(f"Unknown dataset '{name}'. Choices: {DATASET_NAME}")
