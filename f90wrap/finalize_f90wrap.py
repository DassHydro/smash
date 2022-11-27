import numpy as np
import glob
import os

NAME_DT = ["setup", "mesh", "input_data", "parameters", "states", "output"]


def get_pyf90_couple_files(py_mod_names):

    res = set()

    path = "./smash/solver"

    for f90_name, py_name in py_mod_names.items():

        f90_file = glob.glob(f"{path}/**/{f90_name}.f90")[0]
        py_file = f"{path}/{py_name}.py"

        res.add((py_file, f90_file))

    return res


def get_flagged_attr(f90f):
    
    index = set()
    char = set()
    char_array = set()
    private = set()
    
    with open(f90f) as f:

        for l in f:

            if "!>f90w" in l:

                ind_double_2dots = l.find("::") + 2

                subl = l[ind_double_2dots:].strip()

                if "f90w-index" in subl:

                    index.add(subl.split(" ")[0])
                    
                if "f90w-char_array" in subl:
                    
                    char_array.add(subl.split(" ")[0])

                elif "f90w-char" in subl:

                    char.add(subl.split(" ")[0])
                    
                if "f90w-private" in subl:

                    private.add(subl.split(" ")[0])
                    
    res = {
        "index": index,
        "char": char,
        "char_array": char_array,
        "private": private,
    }

    return res


def sed_internal_import(pyf):

    os.system(f'sed -i "0,/import _solver/s//from smash.solver import _solver/" {pyf}')
    os.system(f'sed -i "s/from solver/from smash.solver/g" {pyf}')
    

def sed_private_property(pyf, private):

    for attr in private:

        os.system(f'sed -i "s/def {attr}\\b/def _{attr}/g" {pyf}')
        os.system(f'sed -i "s/@{attr}\\b/@_{attr}/g" {pyf}')
        os.system(f' sed -i "s/self.{attr}\\b/self._{attr}/g" {pyf}')
        os.system(f' sed -i "/ret.append.*{attr}\\b/d" {pyf}')


def sed_index_handler_decorator(pyf, index):
    
    os.system(
        f'sed -i "/from __future__/a \\from smash.solver._f90wrap_decorator import getter_index_handler, setter_index_handler" {pyf}'
    )

    for attr in index:

        os.system(f'sed -i "/def {attr}(self)/i \\\t\@getter_index_handler" {pyf}')
        os.system(f'sed -i "/\\b{attr}.setter/a \\\t\@setter_index_handler" {pyf}')


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
        os.system(f'sed -i "/\\b{attr}.setter/a \\\t\@char_array_setter_handler" {pyf}')


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
                        
                        os.system(
                                f'sed -i "{ind_p}s/ /\\n\tdef copy(self):\\n\t\treturn copy_{name}(self)\\n/" {pyf}'
                            )
                        os.system(
                            f'sed -i "/from __future__/a \\from smash.solver._mw_copy import copy_{name}" {pyf}'
                        )
                    
                        break

def sed_finalise_method(pyf):

    os.system(f'sed -i "/.*_finalise/i \\\t\t\ttry:" {pyf}')
    os.system(f'sed -i "/.*finalise/s/^/\t/" {pyf}')
    os.system(f'sed -i "/.*_finalise/a \\\t\t\texcept:\\n\t\t\t\tpass" {pyf}')


def sed_tab(pyf):

    os.system(f'sed -i "s/\t/    /g" {pyf}')


if __name__ == "__main__":

    py_mod_names_file = "./f90wrap/py_mod_names"
    py_mod_names = eval(open(py_mod_names_file).read())
    
    pyf90_couple_files = get_pyf90_couple_files(py_mod_names)

    for pyf90f in pyf90_couple_files:

        pyf = pyf90f[0]
        f90f = pyf90f[1]

        sed_internal_import(pyf)

        flagged_attr = get_flagged_attr(f90f)
        
        sed_index_handler_decorator(pyf, flagged_attr["index"])

        sed_char_handler_decorator(pyf, flagged_attr["char"])
        
        sed_char_array_handler_decorator(pyf, flagged_attr["char_array"])
        
        sed_private_property(pyf, flagged_attr["private"])

        sed_copy_derived_type(pyf)

        sed_finalise_method(pyf)

        sed_tab(pyf)
