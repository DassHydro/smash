from __future__ import annotations

import smash
import os


__all__ = ["load_dataset"]

DATASET_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_NAME = ["flwdir", "cance", "france"]


def load_dataset(name: str):

    name = name.lower()

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

    else:

        raise ValueError(f"Unknown dataset '{name}'. Choices: {DATASET_NAME}")
