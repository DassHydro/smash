# % ====================================================================
# % .
# % .  SMASH: Spatially distributed Modelling and ASsimilation for Hydrology
# % .
# % .   Portions of this code were written by
# % .     François Colleoni ...
# % .
# % ====================================================================

#% Compiler
FC := gfortran
CC := gcc

#% Compiler flags
F90FLAGS := -cpp -O3 -march=native -funroll-loops -ffast-math -fPIC
debug: F90FLAGS := -Wall -Wextra -fPIC -fmax-errors=1 -cpp -g -fcheck=all -fbacktrace -fcheck-array-temporaries
F77FLAGS := -O3 -march=native -funroll-loops -ffast-math -fPIC
CFLAGS := -g -O3 -march=native -ffast-math -fPIC

#% Files extension
F90EXT := f90
F77EXT := f
CEXT := c
OBJEXT := o

#% Directories
BUILDDIR := obj
SMASHDIR := smash
TAPENADEDIR := tapenade
SOLVERDIR := smash/solver
MESHDIR := smash/mesh

#% INC and MOD for obj
INC := -I$(BUILDDIR)
MOD := -J$(BUILDDIR)

#% f90wrap information
SHAREDLIB := solver
SOLVERMODWRAP := $(SOLVERDIR)/module/mw*_*.f90
OBJWRAP := $(BUILDDIR)/*.o
SOLVERWRAPPERS := f90wrap*.f90

#% Classic `make` call
all: directories c f77 f90 wrappers module meshing finalize library

#% Debug mode `make debug` [Dev]
debug: directories c f77 f90 wrappers module meshing finalize library_edit

#% Making directories
directories:
	@echo "********************************************"
	@echo ""
	@echo " Making directories "
	@echo ""
	@echo "********************************************"
	@mkdir -p $(BUILDDIR)
	@mkdir -p $(SOLVERDIR)/f90wrap/

#% c file(s)
c: \
 obj/adStack.o \
 
#% f77 files
f77: \
 obj/adBuffer.o \
 obj/lbfgsb.o
 
#% f90 files
f90: \
 obj/mwd_common.o \
 obj/mwd_setup.o \
 obj/mwd_mesh.o \
 obj/mwd_input_data.o \
 obj/mwd_parameters.o \
 obj/mwd_states.o \
 obj/mwd_output.o \
 obj/mwd_cost.o \
 obj/md_operator.o \
 obj/mw_run.o \
 obj/mw_adjoint_test.o \
 obj/mw_optimize.o \
 obj/forward.o \
 obj/forward_d.o \
 obj/forward_b.o \
 
#% cpp compile
$(BUILDDIR)/%.$(OBJEXT): $(SOLVERDIR)/*/%.$(CEXT)
	$(CC) $(CFLAGS) $(MOD) $(INC) -c -o $@ $<

#% f77 compile
$(BUILDDIR)/%.$(OBJEXT): $(SOLVERDIR)/*/%.$(F77EXT)
	$(FC) $(F77FLAGS) $(MOD) $(INC) -c -o $@ $<

#% f90 compile
$(BUILDDIR)/%.$(OBJEXT): $(SOLVERDIR)/*/%.$(F90EXT)
	$(FC) $(F90FLAGS) $(MOD) $(INC) -c -o $@ $<
 
#% Making wrappers (f90wrap)
wrappers:
	@echo "********************************************"
	@echo ""
	@echo " Making wrappers "
	@echo ""
	@echo "********************************************"
	f90wrap -m $(SHAREDLIB) $(SOLVERMODWRAP) -k kind_map --package --py-mod-names py_mod_names
	
#% Making module extension (f2py-f90wrap)
module:
	@echo "********************************************"
	@echo ""
	@echo " Making module extension "
	@echo ""
	@echo "********************************************"
	f2py-f90wrap -c --fcompiler=gfortran --f90flags='-cpp -fopenmp -fPIC -fmax-errors=1 -Iobj -Jobj' -lgomp --arch='-march=native' --opt='-O3 -funroll-loops -ffast-math' --build-dir . -m _$(SHAREDLIB) $(OBJWRAP) $(SOLVERWRAPPERS)

#% Making meshing extension (f2py)
meshing:
	@echo "********************************************"
	@echo ""
	@echo " Making meshing extension "
	@echo ""
	@echo "********************************************"
	cd $(MESHDIR) ; python3 -m numpy.f2py -c -m _meshing meshing.f90 skip: mask_upstream_cells downstream_cell_drained_area

#% Making python library (pip3)
library:
	@echo "********************************************"
	@echo ""
	@echo " Making python library "
	@echo ""
	@echo "********************************************"
	pip3 install --compile .

#% Making python library in editable mode (pip3) [Dev]
library_edit:
	@echo "********************************************"
	@echo ""
	@echo " Making python library (editable) "
	@echo ""
	@echo "********************************************"
	pip3 install -e .
	
#% Finalize compilation with mv, rm and sed
finalize:
	mv f90wrap_* $(SOLVERDIR)/f90wrap/.
	mv $(SHAREDLIB)/_mw* $(SOLVERDIR)/.
	mv _$(SHAREDLIB)* $(SOLVERDIR)/.
	rm -rf $(SHAREDLIB)
	bash finalize_f90wrap.sh
	
#% Generating tapenade files (adjoint and tangent linear models)
tap:
	cd $(TAPENADEDIR) ; make
	
#% Clean
clean:
	@$(RM) -rf $(EXTDIR)
	@$(RM) -rf $(BUILDDIR)
	@$(RM) -rf src.*
	@$(RM) -rf *egg-info
	@$(RM) -rf $(SOLVERDIR)/_mw*
	@$(RM) -rf $(SOLVERDIR)/_$(SHAREDLIB)*
	@$(RM) -rf $(SOLVERDIR)/f90wrap
	@$(RM) -rf $(MESHDIR)/*.so
	@$(RM) -rf build
