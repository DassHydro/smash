import smash

import h5py
import pytest
import os

### GLOBAL VARIABLES ###

setup, mesh = smash.load_dataset("Cance")
pytest.model = smash.Model(setup, mesh)

pytest.baseline = h5py.File(
    os.path.join(os.path.dirname(__file__), "baseline.hdf5"), "r"
)
