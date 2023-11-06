import smash
from smash._constant import STRUCTURE

import h5py
import pytest
import os

# TODO: Might add test on France dataset
# TODO: Add parallel tests
### GLOBAL VARIABLES ###

setup, mesh = smash.factory.load_dataset("Cance")

pytest.model = smash.Model(setup, mesh)

setup["sparse_storage"] = True
pytest.sparse_model = smash.Model(setup, mesh)

pytest.model_structure = []
pytest.sparse_model_structure = []

for structure in STRUCTURE:
    (
        setup["snow_module"],
        setup["hydrological_module"],
        setup["routing_module"],
    ) = structure.split("-")
    setup["sparse_storage"] = False
    pytest.model_structure.append(smash.Model(setup, mesh))
    setup["sparse_storage"] = True
    pytest.sparse_model_structure.append(smash.Model(setup, mesh))

pytest.baseline = h5py.File(
    os.path.join(os.path.dirname(__file__), "baseline.hdf5"), "r"
)

pytest.simulated_discharges = h5py.File(
    os.path.join(os.path.dirname(__file__), "simulated_discharges.hdf5"), "r"
)
