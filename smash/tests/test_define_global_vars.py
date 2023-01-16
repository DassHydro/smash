import smash
from h5py import File
import pytest
import os


### GLOBAL VARIABLES will be used during pytest session

setup, mesh = smash.load_dataset("cance")
pytest.model = smash.Model(setup, mesh)

pytest.baseline = File(os.path.join(os.path.dirname(__file__), "baseline.hdf5"))
