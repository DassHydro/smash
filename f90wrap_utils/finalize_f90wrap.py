import numpy as np
import pathlib
import os

NAME_DT = ["setup", "mesh", "input_data", "parameters", "states", "output"]


def get_pyf90_couple_files(py_mod_names: dict) -> list[tuple]:
    """
    Get Python / Fortran file pairs from f90wrap

    Parameters
    ----------
    py_mod_names : dict
        The dictionary where each key, value are the names of the Python
        / Fortran file pairs
    """
    solver_path = pathlib.Path("../smash/solver/")

    res = []

    for f90_name, py_name in py_mod_names.items():
        f90_file = list(solver_path.glob(f"*/{f90_name}.f90"))[0]
        py_file = list(solver_path.glob(f"{py_name}.py"))[0]

        res.append((py_file, f90_file))

    return res


def sed_internal_import(pyf: pathlib.PosixPath):
    """
    Modify internal relative import of the _solver library (.so file)
    to match the package structure. Done by using the unix command sed
    in place

    Parameters
    ----------
    pyf : pathlib.PosixPath
        The Python file to sed
    """

    os.system(f'sed -i "0,/import _solver/s//from smash.solver import _solver/" {pyf}')
    os.system(f'sed -i "s/from solver/from smash.solver/g" {pyf}')


def get_flagged_attr(f90f: pathlib.PosixPath) -> dict[list]:
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
    The decorators are coded in the file smash/solver/_f90wrap_decorator.py

    - index: Allow to switch between 0-based indexing (Python) and 1-based indexing (Fortran).
    It adds one decorator on getter to substract 1 and one decorator on setter to add 1

    - char: Allow to manage Fortran character.
    Fortran derived type character wrapped with f90wrap are retrived as bytes. It adds one
    decorator on getter to decode them and retrived a str

    - char_array: Allow to manage Fortran character array.
    Fortran derived type character array wrapped with f90wrap are retrived as array of ASCII character. It adds
    one decorator on getter to decode ASCII character to str and one decorator on setter to pass numpy character array of list of str
    instead of array of ASCII character

    - private: Allow to make Fortran attribute pseudo-private in Python.
    It adds one "_" at the beginning of each pseudo-private attribute
    """

    index = []
    char = []
    char_array = []
    private = []

    with open(f90f) as f:
        for l in f:
            if "!>f90w" in l:
                ind_double_2dots = l.find("::") + 2

                subl = l[ind_double_2dots:].strip()

                if "f90w-index" in subl:
                    index.append(subl.split(" ")[0])

                if "f90w-char_array" in subl:
                    char_array.append(subl.split(" ")[0])

                elif "f90w-char" in subl:
                    char.append(subl.split(" ")[0])

                if "f90w-private" in subl:
                    private.append(subl.split(" ")[0])

    res = {
        "index": index,
        "char": char,
        "char_array": char_array,
        "private": private,
    }

    return res


def sed_index_handler_decorator(pyf: pathlib.PosixPath, attribute: list[str]):
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
    First, we import the decorator from smash.solver._f90_decorator, then we add
    decorator on top of the getter and setter for each attribute flagged
    """
    os.system(
        f'sed -i "/from __future__/a \\from smash.solver._f90wrap_decorator import getter_index_handler, setter_index_handler" {pyf}'
    )

    for attr in attribute:
        os.system(f'sed -i "/def {attr}(self)/i \\\t\@getter_index_handler" {pyf}')
        os.system(f'sed -i "/\\b{attr}.setter/a \\\t\@setter_index_handler" {pyf}')


def sed_char_handler_decorator(pyf: pathlib.PosixPath, attribute: list[str]):
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
    First, we import the decorator from smash.solver._f90_decorator, then we add
    decorator on top of the getter for each attribute flagged
    """
    os.system(
        f'sed -i "/from __future__/a \\from smash.solver._f90wrap_decorator import char_getter_handler" {pyf}'
    )

    for attr in attribute:
        os.system(f'sed -i "/def {attr}(self)/i \\\t\@char_getter_handler" {pyf}')


def sed_char_array_handler_decorator(pyf: pathlib.PosixPath, attribute: list[str]):
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
    First, we import the decorator from smash.solver._f90_decorator, then we add
    decorator on top of the getter and setter for each attribute flagged
    """
    os.system(
        f'sed -i "/from __future__/a \\from smash.solver._f90wrap_decorator import char_array_getter_handler, char_array_setter_handler" {pyf}'
    )

    for attr in attribute:
        os.system(f'sed -i "/def {attr}(self)/i \\\t\@char_array_getter_handler" {pyf}')
        os.system(f'sed -i "/\\b{attr}.setter/a \\\t\@char_array_setter_handler" {pyf}')


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


# % DEPRECATION: Copy method will be available in a future version of f90wrap
def sed_copy_derived_type(pyf: pathlib.PosixPath):
    name = -1

    for n in NAME_DT:
        if n in pyf.stem.lower():
            name = n

    if name != -1:
        class_name = "class " + name
        ind_cn = np.inf

        with open(pyf) as f:
            for ind_l, l in enumerate(f):
                if class_name in l.lower():
                    ind_cn = ind_l

                if "@property" in l:
                    ind_p = ind_l

                    if ind_p > ind_cn:
                        os.system(
                            f'sed -i "{ind_p}s/ /\\n\tdef copy(self):\\n\t\treturn copy_{name}(self)\\n/" {pyf}'
                        )
                        os.system(
                            f'sed -i "/from __future__/a \\from smash.solver._mw_derived_type_copy import copy_{name}" {pyf}'
                        )

                        break


def sed_finalise_method(pyf: pathlib.PosixPath):
    """
    Add a try / expect condition for each class finalise method.
    Done by using the unix command sed in place
    """
    os.system(f'sed -i "/.*_finalise/i \\\t\t\ttry:" {pyf}')
    os.system(f'sed -i "/.*finalise/s/^/\t/" {pyf}')
    os.system(f'sed -i "/.*_finalise/a \\\t\t\texcept:\\n\t\t\t\tpass" {pyf}')


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

        sed_index_handler_decorator(pyf, flagged_attr["index"])

        sed_char_handler_decorator(pyf, flagged_attr["char"])

        sed_char_array_handler_decorator(pyf, flagged_attr["char_array"])

        sed_private_property(pyf, flagged_attr["private"])

        sed_copy_derived_type(pyf)

        sed_finalise_method(pyf)

        sed_tab(pyf)
