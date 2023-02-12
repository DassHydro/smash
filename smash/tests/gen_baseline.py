from __future__ import annotations

import smash

import argparse
import glob
import importlib
import re
import os
import inspect
import h5py
import pandas as pd
import numpy as np


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "-only",
        "--only",
        nargs="+",
        help="Only tests to include",
    )

    parser.add_argument(
        "-s",
        "-skip",
        "--skip",
        nargs="+",
        help="Skip tests to include",
    )

    args = parser.parse_args()

    if args.only is not None and args.skip is not None:
        parser.error("'--only' and '--skip' options can not be use simultaneously")

    elif args.only:
        args.only = ["generic_" + name for name in args.only]

    elif args.skip:
        args.skip = ["generic_" + name for name in args.skip]

    return args


def adjust_module_names(module_names: list[str]) -> list[str]:
    rep = {"/": ".", ".py": ""}

    rep = dict((re.escape(k), v) for k, v in rep.items())

    pattern = re.compile("|".join(rep.keys()))

    ret = [
        "smash.tests." + pattern.sub(lambda m: rep[re.escape(m.group(0))], name)
        for name in module_names
    ]

    ret.remove("smash.tests.test_define_global_vars")

    return ret


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

    test_name = list()
    status = list()

    for i, key in enumerate(all_keys):
        if len(key) > max_len_name:
            max_len_name = len(key)

        test_name.append(key)

        if key in new_f_keys and key in f_keys:
            if np.array_equal(f[key][:], new_f[key][:]):
                status.append("NON MODIFIED")

            else:
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

    df.to_csv("diff_baseline.csv", sep="|", index=False)

    os.system(f'echo "$(git show)\n\n$(cat diff_baseline.csv)" > diff_baseline.csv')


if __name__ == "__main__":
    args = parser()

    setup, mesh = smash.load_dataset("Cance")
    model = smash.Model(setup, mesh)

    module_names = sorted(glob.glob("**/test_*.py", recursive=True))

    module_names = adjust_module_names(module_names)

    if os.path.exists("new_baseline.hdf5"):
        os.remove("new_baseline.hdf5")

    with h5py.File("new_baseline.hdf5", "w") as f:
        for mn in module_names:
            module = importlib.import_module(mn)

            generic_functions = [
                (name, func)
                for (name, func) in inspect.getmembers(module, inspect.isfunction)
                if name.startswith("generic")
            ]

            if args.only:
                generic_functions = [
                    (name, func)
                    for (name, func) in generic_functions
                    if name in args.only
                ]

            elif args.skip:
                generic_functions = [
                    (name, func)
                    for (name, func) in generic_functions
                    if name not in args.skip
                ]

            for name, func in generic_functions:
                for key, value in func(model=model).items():
                    dump_to_baseline(f, key, value)

    baseline = h5py.File("baseline.hdf5")
    new_baseline = h5py.File("new_baseline.hdf5")

    compare_baseline(baseline, new_baseline)
