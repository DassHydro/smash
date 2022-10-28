import numpy as np
import glob
import os

NAME_DT = ["setup", "mesh", "input_data", "parameters", "states", "output"]


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
    char_array = set()
    private = set()

    with open(f90f) as f:

        for l in f:

            if "!>f90w" in l:

                ind_double_2dots = l.find("::") + 2

                subl = l[ind_double_2dots:].strip()
                
                if "f90w-char_array" in subl:
                    
                    char_array.add(subl.split(" ")[0])

                elif "f90w-char" in subl:

                    char.add(subl.split(" ")[0])

                if "f90w-private" in subl:

                    private.add(subl.split(" ")[0])

    res.update({"char": char, "char_array": char_array, "private": private})

    return res


def sed_internal_import(pyf):

    os.system(f'sed -i "0,/import _solver/s//from smash.solver import _solver/" {pyf}')
    os.system(f'sed -i "s/from solver/from smash.solver/g" {pyf}')


def sed_char_handler_decorator(pyf, char):

    os.system(
        f'sed -i "/from __future__/a \\from smash.solver._f90wrap_decorator import char_getter_handler" {pyf}'
    )

    for attr in char:

        os.system(f'sed -i "/def {attr}(self)/i \\\t\@char_getter_handler" {pyf}')


def sed_char_array_handler_decorator(pyf, char_array):
    
    os.system(
        f'sed -i "/from __future__/a \\from smash.solver._f90wrap_decorator import char_array_getter_handler, char_array_setter_handler" {pyf}'
    )

    for attr in char_array:

        os.system(f'sed -i "/def {attr}(self)/i \\\t\@char_array_getter_handler" {pyf}')
        os.system(f'sed -i "/{attr}.setter/a \\\t\@char_array_setter_handler" {pyf}')
    

def sed_private_property(pyf, private):

    for attr in private:

        os.system(f'sed -i "s/def {attr}/def _{attr}/g" {pyf}')
        os.system(f'sed -i "s/@{attr}/@_{attr}/g" {pyf}')
        os.system(f' sed -i "s/self.{attr}/self._{attr}/g" {pyf}')
        os.system(f' sed -i "/ret.append.*{attr}.*/d" {pyf}')


#% Rework
def sed_copy_derived_type(pyf):
    
    name = -1
    
    for n in NAME_DT:
        
        if n in pyf.lower():
            
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
                        
                        break
                        
        os.system(
            f'sed -i "{ind_p}s/ /\\n\tdef copy(self):\\n\t\treturn copy_{name}(self)\\n/" {pyf}'
        )
        os.system(
            f'sed -i "/from __future__/a \\from smash.solver._mw_routine import copy_{name}" {pyf}'
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

        sed_internal_import(pyf)

        flagged_attr = get_flagged_attr(f90f)

        sed_char_handler_decorator(pyf, flagged_attr["char"])
        
        sed_char_array_handler_decorator(pyf, flagged_attr["char_array"])

        sed_private_property(pyf, flagged_attr["private"])

        sed_copy_derived_type(pyf)

        set_finalise_method(pyf)

        sed_tab(pyf)
