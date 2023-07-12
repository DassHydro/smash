import smash
from smash._constant import STRUCTURE_NAME

import h5py
import pytest
import os

# TODO: Might add test on France dataset
### GLOBAL VARIABLES ###

setup, mesh = smash.factory.load_dataset("Cance")

pytest.model = smash.Model(setup, mesh)

pytest.model_structure = []

for structure in STRUCTURE_NAME:
    setup["structure"] = structure
    pytest.model_structure.append(smash.Model(setup, mesh))

pytest.baseline = h5py.File(
    os.path.join(os.path.dirname(__file__), "baseline.hdf5"), "r"
)
