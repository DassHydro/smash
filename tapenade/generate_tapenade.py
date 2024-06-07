from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
from typing import Callable


class ReplaceRules:
    def __init__(self, pattern: list[str] | None = None, repl: list[str | Callable] | None = None) -> None:
        self.pattern = pattern if pattern is not None else []
        self.repl = repl if repl is not None else []

    def iter(self):
        for i in range(len(self.pattern)):
            yield (self.pattern[i], self.repl[i])

    def __add__(self, other: ReplaceRules):
        return ReplaceRules(self.pattern + other.pattern, self.repl + other.repl)

    def __iadd__(self, other: ReplaceRules):
        self.pattern += other.pattern
        self.repl += other.repl
        return self


def re_replace(file: str, repl_rules: ReplaceRules) -> None:
    assert len(repl_rules.pattern) == len(repl_rules.repl)
    temp_file = f"{file}.tmp"
    with open(temp_file, "w") as tf, open(file, "r") as f:
        buffer = f.read()
        for pattern, repl in repl_rules.iter():
            buffer = re.sub(pattern, repl, buffer, flags=re.DOTALL | re.MULTILINE)
        tf.write(buffer)
    os.replace(temp_file, file)


def patch_fortran_files(fortran_files: list[str], openmp: bool) -> None:
    # Might add other directives
    repl_rules = ReplaceRules([r"(!\$AD start-exclude)(.*?)(!\$AD end-exclude)"], [""])

    if openmp:
        # Only remove directives
        repl_rules += ReplaceRules([r"#ifdef(.*?)_OPENMP", r"#endif"], ["", ""])
    else:
        # Remove everything
        repl_rules += ReplaceRules([r"(#ifdef _OPENMP)(.*?)(#endif)"], [""])

    for f in fortran_files:
        temp_file = os.path.join(".", os.path.basename(f))
        shutil.copy(f, temp_file)
        re_replace(temp_file, repl_rules)


def generate_tapenade_file(fortran_files: list[str], module: str, openmp: bool) -> None:
    tapenade_exe = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "tapenade_3.16", "bin", "tapenade"
    )
    files = [os.path.join(".", os.path.basename(f)) for f in fortran_files]
    cmd_args = [
        tapenade_exe,
        "-b",
        "-d",
        "-fixinterface",
        "-noisize",
        "-context",
        "-msglevel",
        "100",
        "-adjvarname",
        "%_b",
        "-tgtvarname",
        "%_d",
        "-o",
        module,
        "-head",
        r"base_forward_run(parameters.control.x)\(output.cost)",
    ]

    if openmp:
        cmd_args.append("-openmp")

    cmd_args.extend(files)

    subprocess.run(cmd_args, check=True)


def patch_tapenade_file(module: str) -> None:
    repl_rules = ReplaceRules(
        [
            r"TYPE\(PARAMETERSDT_DIFF\)",
            r"TYPE\(RR_PARAMETERSDT_DIFF\)",
            r"TYPE\(RR_STATESDT_DIFF\)",
            r"TYPE\(OUTPUTDT_DIFF\)",
        ],
        [
            "TYPE(PARAMETERSDT)",
            "TYPE(RR_PARAMETERSDT)",
            "TYPE(RR_STATESDT)",
            "TYPE(OUTPUTDT)",
        ],
    )
    tapenade_file = module + "_db.f90"
    re_replace(tapenade_file, repl_rules)


def move_tapenade_file(module: str, build_dir: str) -> None:
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    mv_files = glob.glob("*_db*")
    for f in mv_files:
        shutil.move(f, os.path.join(build_dir, f))

    rm_files = glob.glob("*.f90")
    # Remove filename in case build_dir is the same as the working directory
    filename = module + "_db.f90"
    if module + "_db.f90" in rm_files:
        rm_files.remove(filename)
    for f in rm_files:
        os.remove(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fortran_files",
        nargs="+",
        type=str,
        help="Paths to fortran files in order to generate tapenade file",
    )
    parser.add_argument(
        "-m",
        "--module",
        default="forward",
        type=str,
        help="Name of the module",
        required=False,
    )
    parser.add_argument(
        "--openmp",
        action="store_true",
        default=False,
        help="Use or not OpenMP",
        required=False,
    )
    parser.add_argument(
        "--build-dir",
        default=".",
        type=str,
        help="tapenade generated file is created in this directory",
        required=False,
    )
    args = parser.parse_args()

    patch_fortran_files(args.fortran_files, args.openmp)
    generate_tapenade_file(args.fortran_files, args.module, args.openmp)
    patch_tapenade_file(args.module)
    move_tapenade_file(args.module, args.build_dir)


if __name__ == "__main__":
    main()
