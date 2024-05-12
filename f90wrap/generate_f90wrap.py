#!/usr/bin/env python3

# Note: This file is a wrapper around f90wrap to handle smash specific features and meson build system
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import re
import shutil
import subprocess
from typing import Callable

REPR_METHOD = r"""
def __repr__(self):
    ret = [self.__class__.__name__]
    for attr in dir(self):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(self, attr)
        except Exception:
            continue
        if callable(value):
            continue
        elif isinstance(value, f90wrap.runtime.FortranDerivedTypeArray):
            n = len(value)
            nrepr = 4
            if n == 0:
                continue
            else:
                repr_value = [value[0].__class__.__name__] * min(n, nrepr)
            if n > nrepr:
                repr_value.insert(2, "...")
            repr_value = repr(repr_value)
        else:
            repr_value = repr(value)
        ret.append(f"    {attr}: {repr_value}")
    return "\\n".join(ret)
"""


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


def indent(text: str | None, indents: int = 1) -> str:
    if not text or not isinstance(text, str):
        return ""
    jointext = "".join(["\n"] + ["    "] * indents)
    return jointext.join(text.split("\n"))


def re_replace(file: str, repl_rules: ReplaceRules) -> None:
    assert len(repl_rules.pattern) == len(repl_rules.repl)
    temp_file = f"{file}.tmp"
    with open(temp_file, "w") as tf, open(file, "r") as f:
        buffer = f.read()
        for pattern, repl in repl_rules.iter():
            buffer = re.sub(pattern, repl, buffer, flags=re.DOTALL)
        tf.write(buffer)
    os.replace(temp_file, file)


def get_fortran_flagged_attr(fortran_file: str) -> dict[str, set[str]]:
    flagged_attr = {
        "index": set(),
        "index-array": set(),
        "char": set(),
        "char-array": set(),
        "private": set(),
    }
    with open(fortran_file) as f:
        for line in f:
            # pass commented line
            if line.strip().startswith("!"):
                continue

            if "!$F90W" in line:
                attr_name = re.search(r"::\s*(\w+)", line).group(1)
                f90wrap_flags = re.findall(r"!\$F90W\s*(.*)$", line)[0].split()
                for flag in f90wrap_flags:
                    flagged_attr[flag].add(attr_name)

    return flagged_attr


def get_module_import_replace_rules(module: str) -> ReplaceRules:
    return ReplaceRules(
        [rf"import _{module}", rf"from {module}"], [f"from smash.fcore import _{module}", "from smash.fcore"]
    )


def get_private_attr_replace_rules(attribute: set[str]) -> ReplaceRules:
    repl_rules = ReplaceRules()
    for attr in attribute:
        repl_rules.pattern += [rf"def {attr}\b", rf"\@{attr}\b", rf"self.{attr}\b"]
        repl_rules.repl += [f"def _{attr}", f"@_{attr}", f"self._{attr}"]

    return repl_rules


def get_decorated_attr_replace_rules(flag: str, attribute: set[str]) -> ReplaceRules:
    flag = flag.replace("-", "_")
    repl_rules = ReplaceRules()

    # % Write below from __future__
    repl_rules.pattern += [r"(from __future__ import print_function, absolute_import, division)"]
    repl_rules.repl += [
        rf"\1\nfrom smash.fcore._f90wrap_decorator import f90wrap_getter_{flag}, f90wrap_setter_{flag}"
    ]

    for attr in attribute:
        repl_rules.pattern += [rf"def {attr}\(self\)", rf"(\b{attr}.setter)"]
        repl_rules.repl += [rf"@f90wrap_getter_{flag}\n\t\g<0>", rf"\1\n\t@f90wrap_setter_{flag}"]

    return repl_rules


def get_derived_type_procedure_replace_rules(python_file: str) -> ReplaceRules:
    content = open(python_file, "r").read()
    procedure = {}
    class_matches = re.findall(r"class (\w+)\(f90wrap.runtime.FortranDerivedType\)", content)

    for cm in class_matches:
        pattern = r"def ({0}_\w+\([\w\s,]*\)):".format(cm.lower())
        procedure[cm] = re.findall(pattern, content)

    repl_rules = ReplaceRules()
    for key, value in procedure.items():
        if value:
            for v in value:
                nv = re.search(r"{0}_(.*)".format(key.lower()), v).group(1)
                repl_rules.pattern += [rf"(class {key}\(f90wrap.runtime.FortranDerivedType\):)"]
                repl_rules.repl += [rf"\1\n\tdef {nv}:\n\t\treturn {v}"]
    return repl_rules


def get_del_method_replace_rules() -> ReplaceRules:
    indent_pattern = lambda match: "try:\n\t\t\t" + indent(match.group(0) + "\n\texcept:\n\t\tpass")
    return ReplaceRules(
        [r"if self._alloc(.*?)finalise\(this=self._handle\)"],
        [
            indent_pattern,
        ],
    )


def get_repr_method_replace_rules() -> ReplaceRules:
    return ReplaceRules([r"def __str__\(self\)(.*?)join\(ret\)"], [indent(REPR_METHOD)])


def get_tab_to_space_replace_rules() -> ReplaceRules:
    return ReplaceRules([r"\t"], [4 * " "])


def get_replace_rules(fortran_file: str, python_file: str, module: str) -> ReplaceRules:
    repl_rules = get_module_import_replace_rules(module)

    flagged_attr = get_fortran_flagged_attr(fortran_file)
    for flag, attr in flagged_attr.items():
        if not attr:
            continue
        if flag == "private":
            repl_rules += get_private_attr_replace_rules(attr)
        else:
            repl_rules += get_decorated_attr_replace_rules(flag, attr)

    repl_rules += get_derived_type_procedure_replace_rules(python_file)
    repl_rules += get_del_method_replace_rules()
    repl_rules += get_repr_method_replace_rules()
    repl_rules += get_tab_to_space_replace_rules()

    return repl_rules


def generate_py_mod_names(fortran_files: list[str]) -> None:
    filename = os.path.join("py_mod_names")

    with open(filename, "w") as f:
        f.write("{\n")
        for ff in fortran_files:
            stem = pathlib.Path(ff).stem
            f.write(f"\t'{stem}': '_{stem}',\n")
        f.write("}")


def generate_f90wrap_files(fortran_files: list[str], kind_map_file: str, module: str) -> None:
    cmd_args = [
        "f90wrap",
        "-k",
        kind_map_file,
        "--py-mod-names",
        "py_mod_names",
        "--package",
        "-m",
        module,
    ]

    cmd_args.extend(fortran_files)

    subprocess.run(cmd_args, check=True)


def patch_f90wrap_files(fortran_files: list[str], module: str) -> None:
    py_mod_names = eval(open("py_mod_names").read())

    fpy_couple_file = [
        (ff, os.path.join(module, py_mod_names[pathlib.Path(ff).stem] + ".py")) for ff in fortran_files
    ]

    for ff, pyf in fpy_couple_file:
        re_replace(pyf, get_replace_rules(ff, pyf, module))


def move_f90wrap_files(module: str, build_dir: str) -> None:
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    files = glob.glob("f90wrap*.f90") + glob.glob(os.path.join(module, "*.py"))
    skip_files = [os.path.join(module, "__init__.py")]
    for f in files:
        if f in skip_files:
            continue
        shutil.move(f, os.path.join(build_dir, os.path.basename(f)))

    shutil.rmtree(module)
    os.remove(".f2py_f2cmap")
    os.remove("py_mod_names")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fortran_files",
        nargs="+",
        type=str,
        help="Paths to fortran files in order to generate f90wrap files",
    )
    parser.add_argument(
        "-k",
        "--kind-map-file",
        default="./kind_map",
        type=str,
        help="Path to kind map file",
        required=False,
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
        help="All f90wrap generated files are created in this directory",
        required=False,
    )
    args = parser.parse_args()

    generate_py_mod_names(args.fortran_files)

    generate_f90wrap_files(args.fortran_files, args.kind_map_file, args.module)

    patch_f90wrap_files(args.fortran_files, args.module)

    move_f90wrap_files(args.module, args.build_dir)


if __name__ == "__main__":
    main()
