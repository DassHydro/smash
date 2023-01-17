from __future__ import annotations

import smash

import glob
import importlib
import re
import os
import inspect
import h5py


def adjust_module_names(module_names: list[str]) -> list[str]:

    rep = {"/": ".", ".py": ""}

    rep = dict((re.escape(k), v) for k, v in rep.items())

    pattern = re.compile("|".join(rep.keys()))

    ret = [
        pattern.sub(lambda m: rep[re.escape(m.group(0))], name) for name in module_names
    ]

    ret.remove("test_define_global_vars")

    return ret


def dump_to_baseline(f: h5py.File, key: str, value: np.ndarray):

    f.create_dataset(
        key,
        shape=value.shape,
        dtype=value.dtype,
        data=value,
        compression="gzip",
        chunks=True,
    )


if __name__ == "__main__":

    setup, mesh = smash.load_dataset("Cance")
    model = smash.Model(setup, mesh)

    module_names = sorted(glob.glob("**/test_*.py", recursive=True))

    module_names = adjust_module_names(module_names)

    if os.path.exists("baseline.hdf5"):
        os.remove("baseline.hdf5")

    with h5py.File("baseline.hdf5", "w") as f:

        for mn in module_names:

            module = importlib.import_module(mn)

            generic_functions = [
                (name, func)
                for (name, func) in inspect.getmembers(module, inspect.isfunction)
                if name.startswith("generic")
            ]

            for name, func in generic_functions:
                
                if mn.startswith("core"):

                    for key, value in func(model=model).items():

                        dump_to_baseline(f, key, value)
                    
                elif mn.startswith("mesh"):
                    
                    for key, value in func().items():

                        dump_to_baseline(f, key, value)

                
