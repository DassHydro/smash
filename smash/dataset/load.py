from __future__ import annotations

import smash
import os


__all__ = ["load_dataset"]

DATASET_PATH = os.path.dirname(os.path.realpath(__file__))


def load_dataset(name: str):

    if name == "flwdir":

        return os.path.join(DATASET_PATH, "France_flwdir.tif")

    elif name == "Cance":

        setup = smash.read_setup(os.path.join(DATASET_PATH, "Cance/setup_Cance.yaml"))
        mesh = smash.read_mesh(os.path.join(DATASET_PATH, "Cance/mesh_Cance.hdf5"))

        setup.update(
            {
                "qobs_directory": os.path.join(DATASET_PATH, "Cance/qobs"),
                "prcp_directory": os.path.join(DATASET_PATH, "Cance/prcp"),
                "pet_directory": os.path.join(DATASET_PATH, "Cance/pet"),
            }
        )

        return setup, mesh

    else:

        raise ValueError(f"Unknown dataset '{name}'")
