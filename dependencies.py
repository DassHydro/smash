from __future__ import annotations

import pathlib
import re


def find_ind_module(module_list: list[str], module: str) -> int:
    """
    Find the index of a module in the list of all modules

    Parameters
    ----------
    module_list : list
        The list of all modules

    module : str
        The name of module to find in **module_list**

    Returns
    -------
    res : int
        The index of the **module** in **module_list**
    """
    for i, m in enumerate(module_list):
        if m == module:
            return i
    return -1


def find_ind_files(module_files: list[pathlib.PosixPath], module: str) -> int:
    """
    Find the index of a module in the list of all modules files

    Parameters
    ----------
    module_files : list
        The list of all modules files as pathlib.PosixPath

    module : str
        The name of module to find in **module_list**

    Returns
    -------
    res : int
        The index of the **module** in **module_files**
    """
    for i, m in enumerate(module_files):
        if m.stem == module:
            return i


def find_use_statement(f: pathlib.PosixPath) -> set[str]:
    """
    Function that find Fortran use statement with regex

    Parameters
    ----------
    f : pathlib.PosixPath
        The file to parse as a pathlib.PosixPath object

    Returns
    -------
    res : set
        The set of module used in the file **f**

    Notes
    -----
    An explanation of the pattern: r"^(?!\s*[!]).*?use\s+(\w+)"

    - ^ : Matches the start of a line
    - (?!\s*[!]) : Negative lookahead that excludes matches with ! after zero or more whitespace characters (\s)
    - .*? : Lazy match of any character (.) zero or more times (*) until it encounters the next part of the pattern
    - use\s+ : Matches the string "use" followed by one or more whitespace characters (\s)
    - (\w+) : Captures one or more word characters (\w+) after "use" as a group

    So this regex pattern looks for lines that start with "use" followed by one or more whitespace characters,
    but excludes lines that have a comment symbol (!) before the "use".
    It then captures the module name that follows "use"
    """

    content = open(f, "r").read()
    pattern = r"^(?!\s*[!]).*?use\s+(\w+)"
    ret = [
        el.lower().replace("_diff", "").strip()
        for el in re.findall(
            pattern,
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    ]

    return set(ret)


def get_dependencies(files: list[pathlib.PosixPath]) -> dict[list]:
    """
    Function that find module dependencies for each file

    Parameters
    ----------
    files : list
        The list of file to parse as a pathlib.PosixPath object

    Returns
    -------
    res : dict
        The dictionary of dependencies. The keys are the name of each
        module and the associated values are the set of module dependencies
    """
    ret = {}

    for i, f in enumerate(files):
        deps = find_use_statement(f)

        ret.update({f.stem: deps})

    return ret


def sort_by_dependencies(files: list[pathlib.PosixPath]):
    """
    Function that sort module dependencies for a set of files. The sort
    is performed in place

    Parameters
    ----------
    files : list
        The list of file to sort by dependencies as a pathlib.PosixPath object

    Notes
    -----
    The sorting algorithm is applied sequentially for each module and there is the explanation for one:
    - We look first for it's dependencies and we move the module behind each dependency (Forward loop)
    - We look for dependencies for the current module in other module and we move the current module above each dependency (Backward loop)
    """

    # % Get dependencies dictionary
    deps = get_dependencies(files)
    sorted_deps = list(deps.keys())

    for current_module, use_module in deps.items():
        # % Dependencies found
        if use_module:
            # % Forward
            for um in use_module:
                ind_cm = find_ind_module(sorted_deps, current_module)
                ind_um = find_ind_module(sorted_deps, um)

                # % If use module after current module
                if ind_um > ind_cm:
                    sorted_deps.insert(ind_um + 1, current_module)
                    sorted_deps.pop(ind_cm)

            # % Backward
            for current_module_imd, use_module_imd in deps.items():
                # % If current module in other use module
                if current_module in use_module_imd:
                    ind_cm = find_ind_module(sorted_deps, current_module)
                    ind_cm_imd = find_ind_module(sorted_deps, current_module_imd)

                    # % If current module after other module
                    if ind_cm > ind_cm_imd:
                        sorted_deps.insert(ind_cm + 1, current_module_imd)
                        sorted_deps.pop(ind_cm_imd)

        # % No dependencies, it can be inserted first
        else:
            sorted_deps.remove(current_module)
            sorted_deps.insert(0, current_module)

    # % Finally, sort the file path according to module
    for sd in sorted_deps:
        ind = find_ind_files(files, sd)
        f = files.pop(ind)
        files.insert(len(files), f)


if __name__ == "__main__":
    fcore_path = pathlib.Path("smash/fcore/")

    # % The files are presorted to avoid the randomness that makes the order of dependencies change even if there are no new files.
    # % There can be several ways to sort files by dependencies
    c_files = sorted(list(fcore_path.glob("*/*.c")))
    f77_files = sorted(list(fcore_path.glob("*/*.f")))
    # % In case we have generated f90wrap files we must remove them
    f90_files = sorted(
        list(set(fcore_path.glob("*/*.f90")) - set(fcore_path.glob("f90wrap/*.f90")))
    )

    sort_by_dependencies(f90_files)

    with open("makefile.dep", "w") as f:
        f.write("#% This file is automatically generated by dependencies.py" + 2 * "\n")

        # % c files (currently no dependencies)
        f.write("c: \\" + "\n")
        for cf in c_files:
            f.write(f" obj/{cf.stem}.o \\" + "\n")
        f.write("\n")

        # % f77 files (currently no dependencies)
        f.write("f77: \\" + "\n")
        for f77f in f77_files:
            f.write(f" obj/{f77f.stem}.o \\" + "\n")
        f.write("\n")

        # % f90 files (handle dependencies)
        f.write("f90: \\" + "\n")
        for f90f in f90_files:
            f.write(f" obj/{f90f.stem}.o \\" + "\n")
