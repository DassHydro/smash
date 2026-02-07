from __future__ import annotations

import glob
import importlib
import importlib.util
import inspect
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd

import smash
from smash._constant import HYDROLOGICAL_MODULE, ROUTING_MODULE, SNOW_MODULE

sys.path.insert(0, "")
# Change current directory to smash/smash
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


def adjust_module_names(module_names: list[str]) -> list[str]:
    rep = {"\\": "/"}

    rep = {re.escape(k): v for k, v in rep.items()}

    pattern = re.compile("|".join(rep.keys()))

    ret = [pattern.sub(lambda m: rep[re.escape(m.group(0))], name) for name in module_names]

    if "tests/test_define_global_vars.py" in ret:
        ret.remove("tests/test_define_global_vars.py")

    return ret


def load_module_from_path(path: str):
    abs_path = os.path.abspath(path)
    module_name = "baseline_test_" + re.sub(r"[^a-zA-Z0-9_]", "_", abs_path)
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {abs_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump_to_baseline(f: h5py.File, key: str, value: np.ndarray):
    if value.dtype.char == "U":
        value = value.astype("S")

    f.create_dataset(
        key,
        shape=value.shape,
        dtype=value.dtype,
        data=value,
        compression="gzip",
        chunks=True,
    )


def compare_baseline(f: h5py.File, new_f: h5py.File):
    max_len_name = 0

    new_f_keys = list(new_f.keys())
    f_keys = list(f.keys())

    all_keys = sorted(set(new_f_keys + f_keys))

    test_name = []
    status = []

    for key in all_keys:
        max_len_name = max(len(key), max_len_name)

        test_name.append(key)

        if key in new_f_keys and key in f_keys:
            # % If an error occurs during check (inconsistent shapes ...)
            try:
                if f[key][:].size == 0:
                    is_equal = new_f[key][:].size == 0
                elif f[key][:].dtype == "object" or f[key][:].dtype.char == "S":
                    is_equal = np.array_equal(f[key][:], new_f[key][:])
                else:
                    is_equal = np.allclose(f[key][:], new_f[key][:], equal_nan=True, atol=1e-3)

                if is_equal:
                    status.append("NON MODIFIED")

                else:
                    status.append("MODIFIED")
            except Exception:
                status.append("MODIFIED")

            new_f_keys.remove(key)
            f_keys.remove(key)

        elif key in f_keys:
            status.append("DELETED")
            f_keys.remove(key)

        else:
            status.append("ADDED")
            new_f_keys.remove(key)

    test_name = [tn + (max_len_name - len(tn) + 1) * " " for tn in test_name]

    df = pd.DataFrame()

    df["TEST NAME" + (max_len_name - 8) * " "] = test_name
    df["STATUS"] = status

    df.to_csv("tests/diff_baseline.csv", sep="|", index=False)

    os.system('echo "$(git show --no-patch)\n\n$(cat tests/diff_baseline.csv)" > tests/diff_baseline.csv')


if __name__ == "__main__":
    # % Disable stderr
    # % Disable tqdm progress bar (printed to standard error)
    sys.stderr = open("/dev/null", "w")

    setup, mesh = smash.factory.load_dataset("Cance")

    print("collecting ...")

    qs = h5py.File(os.path.join(os.path.dirname(__file__), "simulated_discharges.hdf5"), "r")["sim_q"][:]

    # % Disable stdout
    # % TODO: replace this by adding a verbose argument at Model initialisation
    sys.stdout = open("/dev/null", "w")

    model = smash.Model(setup, mesh)

    # Do not need to read prcp and pet again
    setup["read_prcp"] = False
    setup["read_pet"] = False
    model_structure = []

    # Select base structure
    base_sm, base_hm, base_rm = "zero", "gr4", "lr"
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
        wmodel = smash.Model(setup, mesh)
        wmodel.atmos_data.prcp = model.atmos_data.prcp
        wmodel.atmos_data.pet = model.atmos_data.pet
        if "ci" in wmodel.rr_parameters.keys:
            wmodel.set_rr_parameters("ci", model.get_rr_parameters("ci"))
        model_structure.append(wmodel)

    # # % Enable stdout
    sys.stdout = sys.__stdout__
    module_names = sorted(glob.glob("tests/**/test_*.py", recursive=True))

    module_names = adjust_module_names(module_names)

    if os.path.exists("tests/new_baseline.hdf5"):
        os.remove("tests/new_baseline.hdf5")

    with h5py.File("tests/new_baseline.hdf5", "w") as f:
        for mn in module_names:
            print(mn, end=" ")
            module = load_module_from_path(mn)

            generic_functions = [
                (name, func)
                for (name, func) in inspect.getmembers(module, inspect.isfunction)
                if name.startswith("generic")
            ]

            for _, func in generic_functions:
                for key, value in func(
                    model=model,
                    model_structure=model_structure,
                    qs=qs,
                ).items():
                    dump_to_baseline(f, key, value)
                print(".", end="", flush=True)
            print("")

    baseline = h5py.File("tests/baseline.hdf5")
    new_baseline = h5py.File("tests/new_baseline.hdf5")

    compare_baseline(baseline, new_baseline)

    # % Enable stderr
    sys.stderr = sys.__stderr__
