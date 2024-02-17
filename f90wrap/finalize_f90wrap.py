from __future__ import annotations

import os
import pathlib
import re

REPR_METHOD = """
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
    return "%(new_line)s".join(ret)
"""


def indent(text: str | None, indents: int = 1) -> str:
    if not text or not isinstance(text, str):
        return ""
    jointext = "".join(["\n"] + ["    "] * indents)
    return jointext.join(text.split("\n"))


def get_pyf90_couple_files(py_mod_names: dict[str, str]) -> list[tuple[str, str]]:
    """
    Get Python / Fortran file pairs from f90wrap

    Parameters
    ----------
    py_mod_names : dict
        The dictionary where each key, value are the names of the Python
        / Fortran file pairs
    """
    fcore_path = pathlib.Path("../smash/fcore/")

    res = []

    for f90_name, py_name in py_mod_names.items():
        f90_file = next(iter(fcore_path.glob(f"*/{f90_name}.f90")))
        py_file = next(iter(fcore_path.glob(f"{py_name}.py")))

        res.append((py_file, f90_file))

    return res


def sed_internal_import(pyf: pathlib.PosixPath):
    """
    Modify internal relative import of the _libfcore library (.so file)
    to match the package structure. Done by using the unix command sed
    in place

    Parameters
    ----------
    pyf : pathlib.PosixPath
        The Python file to sed
    """

    os.system(f'sed -i "0,/import _libfcore/s//from smash.fcore import _libfcore/" {pyf}')
    os.system(f'sed -i "s/from libfcore/from smash.fcore/g" {pyf}')


def get_flagged_attr(f90f: pathlib.PosixPath) -> dict[str, list[str]]:
    """
    Get the flagged derived type attributes in Fortran

    Parameters
    ----------
    f90f : pathlib.PosixPath
        The Fortran file to parse

    Returns
    -------
    res : dict
        A dictionary where each key is a flag and value the derived type
        attribute flagged

    Notes
    -----
    Currently, there are 4 flags on Fortran derived type attributers
    which allow to add specific decorators on f90wrap generated Python files.
    The decorators are coded in the file smash/fcore/_f90wrap_decorator.py

    - index: Allow to switch between 0-based indexing (Python) and 1-based indexing (Fortran).
    It adds one decorator on getter to substract 1 and one decorator on setter to add 1

    - index-array: Allow to switch between 0-based indexing (Python) and 1-based indexing (Fortran).
    Same as **index** but for arrays

    - char: Allow to manage Fortran character.
    Fortran derived type character wrapped with f90wrap are retrived as bytes. It adds one
    decorator on getter to decode them and retrived a str

    - char-array: Allow to manage Fortran character array.
    Fortran derived type character array wrapped with f90wrap are retrived as array of ASCII character. It
    adds one decorator on getter to decode ASCII character to str and one decorator on setter to pass numpy
    character array of list of str
    instead of array of ASCII character

    - private: Allow to make Fortran attribute pseudo-private in Python.
    It adds one "_" at the beginning of each pseudo-private attribute
    """

    index = []
    index_array = []
    char = []
    char_array = []
    private = []

    with open(f90f) as f:
        for line in f:
            if "!$F90W" in line:
                ind_double_2dots = line.find("::") + 2

                subline = line[ind_double_2dots:].strip().lower()

                if "index-array" in subline:
                    index_array.append(subline.split(" ")[0])

                elif "index" in subline:
                    index.append(subline.split(" ")[0])

                if "char-array" in subline:
                    char_array.append(subline.split(" ")[0])

                elif "char" in subline:
                    char.append(subline.split(" ")[0])

                if "private" in subline:
                    private.append(subline.split(" ")[0])

    res = {
        "index": index,
        "index-array": index_array,
        "char": char,
        "char-array": char_array,
        "private": private,
    }

    return res


def sed_index_decorator(pyf: pathlib.PosixPath, attribute: list[str]):
    """
    Modify Python script to handle index decorator for specific attributes.
    Done by using the unix command sed in place

    Parameters
    ----------
    pyf : pathlib.PosixPath
        The Python file to sed

    attribute: list
        The list of attributes to sed

    Notes
    -----
    First, we import the decorator from smash.fcore._f90_decorator, then we add
    decorator on top of the getter and setter for each attribute flagged
    """
    os.system(
        f'sed -i "/from __future__/a \\from smash.fcore._f90wrap_decorator import f90wrap_getter_index, '
        f'f90wrap_setter_index" {pyf}'
    )

    for attr in attribute:
        os.system(f'sed -i "/def {attr}(self)/i \\\t\\@f90wrap_getter_index" {pyf}')
        os.system(f'sed -i "/\\b{attr}.setter/a \\\t\\@f90wrap_setter_index" {pyf}')


def sed_index_array_decorator(pyf: pathlib.PosixPath, attribute: list[str]):
    """
    Modify Python script to handle index array decorator for specific attributes.
    Done by using the unix command sed in place

    Parameters
    ----------
    pyf : pathlib.PosixPath
        The Python file to sed

    attribute: list
        The list of attributes to sed

    Notes
    -----
    First, we import the decorator from smash.fcore._f90_decorator, then we add
    decorator on top of the getter and setter for each attribute flagged
    """
    os.system(
        f'sed -i "/from __future__/a \\from smash.fcore._f90wrap_decorator import '
        f'f90wrap_getter_index_array, f90wrap_setter_index_array" {pyf}'
    )

    for attr in attribute:
        os.system(f'sed -i "/def {attr}(self)/i \\\t\\@f90wrap_getter_index_array" {pyf}')
        os.system(f'sed -i "/\\b{attr}.setter/a \\\t\\@f90wrap_setter_index_array" {pyf}')


def sed_char_decorator(pyf: pathlib.PosixPath, attribute: list[str]):
    """
    Modify Python script to handle character decorator for specific attributes.
    Done by using the unix command sed in place

    Parameters
    ----------
    pyf : pathlib.PosixPath
        The Python file to sed

    attribute: list
        The list of attributes to sed

    Notes
    -----
    First, we import the decorator from smash.fcore._f90_decorator, then we add
    decorator on top of the getter for each attribute flagged
    """
    os.system(
        f'sed -i "/from __future__/a \\from smash.fcore._f90wrap_decorator import f90wrap_getter_char" {pyf}'
    )

    for attr in attribute:
        os.system(f'sed -i "/def {attr}(self)/i \\\t\\@f90wrap_getter_char" {pyf}')


def sed_char_array_decorator(pyf: pathlib.PosixPath, attribute: list[str]):
    """
    Modify Python script to handle character array decorator for specific attributes.
    Done by using the unix command sed in place

    Parameters
    ----------
    pyf : pathlib.PosixPath
        The Python file to sed

    attribute: list
        The list of attributes to sed.

    Notes
    -----
    First, we import the decorator from smash.fcore._f90_decorator, then we add
    decorator on top of the getter and setter for each attribute flagged
    """
    os.system(
        f'sed -i "/from __future__/a \\from smash.fcore._f90wrap_decorator import '
        f'f90wrap_getter_char_array, f90wrap_setter_char_array" {pyf}'
    )

    for attr in attribute:
        os.system(f'sed -i "/def {attr}(self)/i \\\t\\@f90wrap_getter_char_array" {pyf}')
        os.system(f'sed -i "/\\b{attr}.setter/a \\\t\\@f90wrap_setter_char_array" {pyf}')


def sed_private_property(pyf: pathlib.PosixPath, attribute: list[str]):
    """
    Modify Python script make pseudo-private property for specific attributes.
    Done by using the unix command sed in place

    Parameters
    ----------
    pyf : pathlib.PosixPath
        The Python file to sed

    attribute: list
        The list of attributes to sed.

    Notes
    -----
    We add a "_" each time we encouter the attribute to modify
    """

    for attr in attribute:
        os.system(f'sed -i "s/def {attr}\\b/def _{attr}/g" {pyf}')
        os.system(f'sed -i "s/@{attr}\\b/@_{attr}/g" {pyf}')
        os.system(f'sed -i "s/self.{attr}\\b/self._{attr}/g" {pyf}')
        os.system(f'sed -i "/ret.append.*{attr}\\b/d" {pyf}')


def sed_derived_type_procedure(pyf: pathlib.PosixPath):
    """
    Add to class method Fortran type-bound procedure.
    """

    content = open(pyf, "r").read()
    procedure = {}
    class_matches = re.findall('@f90wrap.runtime.register_class\\("libfcore.(\\w+)', content)

    for cm in class_matches:
        pattern = r"def ({0}_\w+\([\w\s,]*\)):".format(cm.lower())
        procedure[cm] = re.findall(pattern, content)

    for key, value in procedure.items():
        if value:
            pattern = f"class {key}(f90wrap.runtime.FortranDerivedType)"
            for vle in value:
                nvle = re.search(r"{0}_(.*)".format(key.lower()), vle).group(1)
                os.system(f'sed -i "/{pattern}/a \\\tdef {nvle}:\\n\\t\\treturn {vle}" {pyf}')


def sed_finalise_method(pyf: pathlib.PosixPath):
    """
    Add a try / expect condition for each class finalise method.
    Done by using the unix command sed in place
    """
    os.system(f'sed -i "/.*_finalise/i \\\t\t\ttry:" {pyf}')
    os.system(f'sed -i "/.*finalise/s/^/\t/" {pyf}')
    os.system(f'sed -i "/.*_finalise/a \\\t\t\texcept:\\n\t\t\t\tpass" {pyf}')


def sed_repr_method(pyf: pathlib.PosixPath):
    """
    Replace __str__ method with __repr__. It allows to display
    object without calling print explicitly. The function is overwrited
    by REPR_METHOD. This is a monkey patch of f90wrap.
    Done by using the unix command sed in place
    """

    n_indents = 1

    # indent start
    repr_method = "\\\t" * n_indents + REPR_METHOD.strip()

    # indent each new line
    repr_method = indent(repr_method, indents=n_indents)

    # replace new lines by raw new lines
    repr_method = repr_method.replace("\n", r"\n")
    repr_method = repr_method.replace("%(new_line)s", r"\\n")

    # Sed command that removes lines between def _str__ and join(ret) and replaces by REPR_METHOD
    os.system(f"sed -i '/def __str__(self):/,/join(ret)/c{repr_method}' {pyf}")


def sed_tab(pyf: pathlib.PosixPath):
    """
    Replace tabs with 4 spaces.
    Done by using the unix command sed in place
    """
    os.system(f'sed -i "s/\t/    /g" {pyf}')


if __name__ == "__main__":
    py_mod_names = eval(open("py_mod_names").read())

    pyf90_couple_files = get_pyf90_couple_files(py_mod_names)

    for pyf, f90f in pyf90_couple_files:
        sed_internal_import(pyf)

        flagged_attr = get_flagged_attr(f90f)

        sed_index_decorator(pyf, flagged_attr["index"])

        sed_index_array_decorator(pyf, flagged_attr["index-array"])

        sed_char_decorator(pyf, flagged_attr["char"])

        sed_char_array_decorator(pyf, flagged_attr["char-array"])

        sed_private_property(pyf, flagged_attr["private"])

        sed_derived_type_procedure(pyf)

        sed_finalise_method(pyf)

        sed_repr_method(pyf)

        sed_tab(pyf)
