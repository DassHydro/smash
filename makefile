#% Compiler
FC := gfortran
CC := gcc

#% Compiler flags
F90_FLAGS := -cpp -O3 -march=native -funroll-loops -fPIC -fopenmp
dbg: F90_FLAGS := -Wall -Wextra -fPIC -fopenmp -fmax-errors=1 -cpp -g -fcheck=all -fbacktrace -fcheck-array-temporaries
F77_FLAGS := -O3 -march=native -funroll-loops -fPIC -fopenmp
C_FLAGS := -O3 -march=native -fPIC -fopenmp

#% Files extension
F90_EXT := f90
F77_EXT := f
C_EXT := c
OBJ_EXT := o

#% Directories
BUILD_DIR := obj
SMASH_DIR := smash
TAPENADE_DIR := tapenade
F90WRAP_DIR := f90wrap
FCORE_DIR := smash/fcore
MESH_DIR := smash/factory/mesh
TESTS_DIR := smash/tests
DOC_DIR := doc

#% INC and MOD for obj
INC := -I$(BUILD_DIR)
MOD := -J$(BUILD_DIR)

#% Files
OBJ_FILES := $(BUILD_DIR)/*.$(OBJ_EXT)
FCORE_FILES := $(FCORE_DIR)/*/*.$(C_EXT) $(FCORE_DIR)/*/*.$(F77_EXT) $(FCORE_DIR)/*/*.$(F90_EXT)
FCORE_WRAP_FILES := $(FCORE_DIR)/*/mwd_*.$(F90_EXT) $(FCORE_DIR)/*/mw_*.$(F90_EXT)
FCORE_F90WRAP_FILES := f90wrap*.$(F90_EXT)
MESH_WRAP_FILES := $(MESH_DIR)/mw_*.$(F90_EXT)

#% Shared lib name
FCORE_SHARED_LIB := libfcore
MESH_SHARED_LIB := libmesh

#% Classic `make` call
#% 'c' 'f77' and 'f90' targets are in makefile.dep
all: directories c f77 f90 f90wrap f2py-f90wrap f2py-mesh finalize library

#% Debug mode `make debug` [Dev]
#% 'c' 'f77' and 'f90' targets are in makefile.dep
dbg: directories c f77 f90 f90wrap f2py-f90wrap f2py-mesh finalize library-edit

#% Make directories
directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(FCORE_DIR)/f90wrap/

#% Make dependencies
#% Automatic generation of makefile.dep file (allows not to modify by hand the order of dependencies of the files to compile)
dependencies:
	python3 dependencies.py

#% Compile c
$(BUILD_DIR)/%.$(OBJ_EXT): $(FCORE_DIR)/*/%.$(C_EXT)
	$(CC) $(C_FLAGS) $(MOD) $(INC) -c -o $@ $<

#% Compile f77
$(BUILD_DIR)/%.$(OBJ_EXT): $(FCORE_DIR)/*/%.$(F77_EXT)
	$(FC) $(F77_FLAGS) $(MOD) $(INC) -c -o $@ $<

#% Compile f90
$(BUILD_DIR)/%.$(OBJ_EXT): $(FCORE_DIR)/*/%.$(F90_EXT)
	$(FC) $(F90_FLAGS) $(MOD) $(INC) -c -o $@ $<

#% Make f90wrap files
#% Automatic generation of py_mod_names file (allow to do not modified this file each time a wrapped module is added to source)
f90wrap:
	rm -rf $(BUILD_DIR)/f90wrap*
	cd $(F90WRAP_DIR) ; python3 gen_py_mod_names.py
	f90wrap -m $(FCORE_SHARED_LIB) $(FCORE_WRAP_FILES) -k $(F90WRAP_DIR)/kind_map --py-mod-names $(F90WRAP_DIR)/py_mod_names --package

#% Make f2py-f90wrap .so library
f2py-f90wrap:
	f2py-f90wrap -c --fcompiler=gfortran --f90flags='-cpp -fopenmp -fPIC -fmax-errors=1 -Iobj -Jobj' -lgomp --arch='-march=native' --opt='-O3 -funroll-loops' --build-dir . -m _$(FCORE_SHARED_LIB) $(OBJ_FILES) $(FCORE_F90WRAP_FILES)

#% Make f2py-meshing .so library
f2py-mesh:
	python3 -m numpy.f2py -c -m _$(MESH_SHARED_LIB) $(MESH_WRAP_FILES) skip: mask_upstream_cells fill_nipd downstream_cell_flwacc argsort_i argsort_r

#% Make python library (pip3)
library:
	pip3 install --compile .

#% Make python library in editable mode (pip3) [Dev]
library-edit:
	pip3 install -e .

#% Finalize compilation with mv, rm and sed
finalize:
	mv f90wrap_*.f90 $(FCORE_DIR)/f90wrap/.
	mv f90wrap_*.o $(BUILD_DIR)/.
	mv $(FCORE_SHARED_LIB)/_mw* $(FCORE_DIR)/.
	mv _$(FCORE_SHARED_LIB)* $(FCORE_DIR)/.
	mv _$(MESH_SHARED_LIB)* $(MESH_DIR)/.
	rm -rf $(FCORE_SHARED_LIB)
	cd $(F90WRAP_DIR) ; python3 finalize_f90wrap.py

#% Generate tapenade file (adjoint and tangent linear models)
tap:
	cd $(TAPENADE_DIR) ; make

#% Compare tapenade file
tap-cmp:
	cd $(TAPENADE_DIR) ; make tap-cmp

#% Generate sphinx documentation
doc:
	cd $(DOC_DIR) ; make html

#% Clean sphinx documentation
doc-clean:
	cd $(DOC_DIR) ; make clean

#% Testing code with pytest
test:
	cd $(TESTS_DIR) ; pytest

#% Testing code with pytest and coverage
test-coverage:
	cd $(TESTS_DIR) ; pytest --cov-report term --cov-report html --cov=smash

#% Generate baseline for test with args (see argparser in gen_baseline.py)
test-baseline:
	cd $(TESTS_DIR) ; python3 gen_baseline.py $(args)

#% Clean
clean:
	@rm -rf $(EXT_DIR)
	@rm -rf $(BUILD_DIR)
	@rm -rf src.*
	@rm -rf *egg-info
	@rm -rf $(FCORE_DIR)/_mw*
	@rm -rf $(FCORE_DIR)/_$(FCORE_SHARED_LIB)*
	@rm -rf $(FCORE_DIR)/f90wrap
	@rm -rf $(MESH_DIR)/_$(MESH_SHARED_LIB)*.so
	@rm -rf build

.PHONY: all dbg directories dependencies c f77 f90 f90wrap f2py-f90wrap f2py-mesh library library-edit finalize tap tap-cmp doc doc-clean test test-baseline clean

#% Include 'c' 'f77' and 'f90' targets
include makefile.dep
