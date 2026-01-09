import os

import h5py
import pytest

import smash
from smash._constant import HYDROLOGICAL_MODULE, ROUTING_MODULE, SNOW_MODULE

AVAILABLE_TEST_LEVEL = ("light", "full")

# TODO: Might add test on France dataset
# TODO: Add parallel tests
### GLOBAL VARIABLES ###

test_level = os.environ.get("SMASH_TEST_LEVEL") or "full"
assert test_level in AVAILABLE_TEST_LEVEL, (
    f"Unknown test_level '{test_level}'. Choices: {AVAILABLE_TEST_LEVEL}"
)

setup, mesh = smash.factory.load_dataset("Cance")

pytest.model = smash.Model(setup, mesh)

setup["sparse_storage"] = True
pytest.sparse_model = smash.Model(setup, mesh)

# Do not need to read prcp and pet again
setup["read_prcp"] = False
setup["read_pet"] = False
pytest.model_structure = []
pytest.sparse_model_structure = []

# Select base structure
base_sm, base_hm, base_rm = "zero", "gr4", "lr"
if test_level == "light":
    structure = [f"{base_sm}-{base_hm}-{base_rm}"]
elif test_level == "full":
    structure = []
    structure.extend([f"{sm}-{base_hm}-{base_rm}" for sm in SNOW_MODULE])
    structure.extend([f"{base_sm}-{hm}-{base_rm}" for hm in HYDROLOGICAL_MODULE])
    structure.extend([f"{base_sm}-{base_hm}-{rm}" for rm in ROUTING_MODULE])

for struct in structure:
    (
        setup["snow_module"],
        setup["hydrological_module"],
        setup["routing_module"],
    ) = struct.split("-")

    setup["sparse_storage"] = False
    model = smash.Model(setup, mesh)
    model.atmos_data.prcp = pytest.model.atmos_data.prcp
    model.atmos_data.pet = pytest.model.atmos_data.pet

    setup["sparse_storage"] = True
    sparse_model = smash.Model(setup, mesh)
    for i in range(sparse_model.setup.ntime_step):
        sparse_model.atmos_data.sparse_prcp[i] = pytest.sparse_model.atmos_data.sparse_prcp[i]
        sparse_model.atmos_data.sparse_pet[i] = pytest.sparse_model.atmos_data.sparse_pet[i]

    if "ci" in model.rr_parameters.keys:
        model.set_rr_parameters("ci", pytest.model.get_rr_parameters("ci"))
        sparse_model.set_rr_parameters("ci", pytest.sparse_model.get_rr_parameters("ci"))

    pytest.model_structure.append(model)
    pytest.sparse_model_structure.append(sparse_model)

pytest.baseline = h5py.File(os.path.join(os.path.dirname(__file__), "baseline.hdf5"), "r")

pytest.simulated_discharges = h5py.File(
    os.path.join(os.path.dirname(__file__), "simulated_discharges.hdf5"), "r"
)
