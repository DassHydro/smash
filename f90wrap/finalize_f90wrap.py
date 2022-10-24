import numpy as np
import glob
import os

NAME_DT = ["setup", "mesh", "input_data", "parameters", "states", "output"]
NAME_DT_CLASS = ["class " + n for n in NAME_DT]


def get_py_mod_names(py_mod_names_file):

    res = set()

    with open(py_mod_name_file) as f:

        for l in f:

            ind_2dots = l.find(":")

            if ind_2dots != -1:

                subl = l[0:ind_2dots].strip()

                subl = subl.replace('"', "")

                res.add(subl)

    return res


def get_pyf90_couple_files(py_mod_name):

    res = set()

    py_path = "./smash/solver/"
    f90_path = "./smash/solver/module/"

    for name in py_mod_name:

        py_file = py_path + "_" + name + ".py"
        f90_file = f90_path + name + ".f90"

        res.add((py_file, f90_file))

    return res


def get_flagged_attr(f90f):

    res = {}

    char = set()
    private = set()

    with open(f90f) as f:

        for l in f:

            if "!>f90w" in l:

                ind_double_dot = l.find("::") + 2

                subl = l[ind_double_dot:].strip()

                if "f90w-char" in subl:

                    char.add(subl.split(" ")[0])

                if "f90w-private" in subl:

                    private.add(subl.split(" ")[0])

    res.update({"char": char, "private": private})

    return res


def sed_external_package(pyf):

    os.system(f'sed -i "/import logging/a \import numpy" {pyf}')


def sed_internal_import(pyf):

    os.system(f'sed -i "0,/import _solver/s//from smash.solver import _solver/" {pyf}')
    os.system(f'sed -i "s/from solver/from smash.solver/g" {pyf}')


def sed_char_handler_decorator(pyf, char):

    os.system(
        f'sed -i "/from __future__/a \\from smash.solver.\_f90wrap_decorator import character_handler" {pyf}'
    )

    for attr in char:

        os.system(f'sed -i "/def {attr}/i \\\t\@character_handler" {pyf}')


def sed_private_property(pyf, private):

    for attr in private:

        os.system(f'sed -i "s/def {attr}/def _{attr}/g" {pyf}')
        os.system(f'sed -i "s/@{attr}/@_{attr}/g" {pyf}')
        os.system(f' sed -i "s/self.{attr}/self._{attr}/g" {pyf}')
        os.system(f' sed -i "/ret.append.*{attr}.*/d" {pyf}')


def sed_copy_derived_type(pyf):

    ind_nc_save = -1
    ind_l_save = -1

    with open(pyf) as f:

        for ind_l, l in enumerate(f):

            for ind_nc, nc in enumerate(NAME_DT_CLASS):

                if nc in l.lower():

                    ind_nc_save = ind_nc

            if "@property" in l:

                ind_l_save = ind_l

                break

    if ind_nc_save != -1 and ind_l_save != -1:

        copy_name = NAME_DT[ind_nc_save]
        os.system(
            f'sed -i "{ind_l_save}s/ /\\n\tdef copy(self):\\n\t\treturn copy_{copy_name}(self)\\n/" {pyf}'
        )
        os.system(
            f'sed -i "/from __future__/a \\from smash.solver._mw_routine import copy_{copy_name}" {pyf}'
        )


def set_finalise_method(pyf):

    os.system(f'sed -i "/.*_finalise/i \\\t\t\ttry:" {pyf}')
    os.system(f'sed -i "/.*finalise/s/^/\t/" {pyf}')
    os.system(f'sed -i "/.*_finalise/a \\\t\t\texcept:\\n\t\t\t\tpass" {pyf}')


def sed_tab(pyf):

    os.system(f'sed -i "s/\t/    /g" {pyf}')


if __name__ == "__main__":

    py_mod_name_file = "./f90wrap/py_mod_names"
    py_mod_name = get_py_mod_names(py_mod_name_file)

    pyf90_couple_files = get_pyf90_couple_files(py_mod_name)

    for pyf90f in pyf90_couple_files:

        pyf = pyf90f[0]
        f90f = pyf90f[1]

        sed_external_package(pyf)

        sed_internal_import(pyf)

        flagged_attr = get_flagged_attr(f90f)

        sed_char_handler_decorator(pyf, flagged_attr["char"])

        sed_private_property(pyf, flagged_attr["private"])

        sed_copy_derived_type(pyf)

        set_finalise_method(pyf)

        sed_tab(pyf)
