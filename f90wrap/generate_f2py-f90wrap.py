#!/usr/bin/env python3

# Note: This file is a wrapper around f90wrap to handle smash specific features and meson build system
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


def generate_f2py_f90wrap_files(fortran_files: list[str], module: str, build_dir: str) -> None:
    cmd_args = [
        "f2py-f90wrap",
        "-m",
        module,
        "--lower",
        "--build-dir",
        build_dir,
    ]

    cmd_args.extend(fortran_files)

    subprocess.run(cmd_args, check=True)


def patch_c_file(module: str, build_dir: str) -> None:
    # Patch the generated C file to replace setjmpex with setjmp header. This is necessary to build on
    # Windows 2022 (and later).
    if sys.platform == "win32":
        libfile = os.path.join(build_dir, f"{module}module.c")
        with open(libfile, "r+") as f:
            buffer = f.read()
            buffer = re.sub("setjmpex.h", "setjmp.h", buffer, flags=re.DOTALL)
            f.seek(0)
            f.write(buffer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fortran_files",
        nargs="+",
        type=str,
        help="Paths to fortran files in order to generate f2py-f90wrap files",
    )
    parser.add_argument(
        "-m",
        "--module",
        default="mod",
        type=str,
        help="Name of the module",
        required=False,
    )
    parser.add_argument(
        "--build-dir",
        default=".",
        type=str,
        help="All f2py-f90wrap generated files are created in this directory",
        required=False,
    )
    args = parser.parse_args()

    generate_f2py_f90wrap_files(args.fortran_files, args.module, args.build_dir)

    patch_c_file(args.module, args.build_dir)


if __name__ == "__main__":
    main()
