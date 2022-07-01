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

#% F90WRAP information
SHAREDLIB := solver
SOLVERMODWRAP := $(SOLVERSRC)/module/mw_*.f90
OBJWRAP := $(BUILDDIR)/*.o
SOLVERWRAPPERS := f90wrap*.f90

#% Classic `make` call
all: directories cpp f77 f90 wrappers module meshing finalize

#% Debug mode `make debug` (Developper)
debug: directories cpp f77 f90 wrappers module meshing finalize

#% Making directories
directories:
	@echo "********************************************"
	@echo ""
	@echo " Making directories "
	@echo ""
	@echo "********************************************"
	@mkdir -p $(BUILDDIR)
	@mkdir -p $(SOLVERSRC)/f90wrap/

#% cpp file(s)
cpp: \
 obj/adStack.o \
 
#% f77 files
f77: \
 obj/adBuffer.o \
 obj/lbfgsb.o
 
#% f90 files
f90: \
 obj/m_common.o \
 obj/mw_setup.o \
 obj/mw_mesh.o \
 obj/mw_input_data.o \
 obj/mw_parameters.o \
 obj/mw_states.o \
 obj/mw_output.o \
 obj/mw_cost.o \
 obj/m_operator.o \
 obj/mw_run.o \
 obj/mw_validate.o \
 obj/mw_optimize.o \
 obj/mw_utils.o \
 obj/forward.o \
 obj/forward_d.o \
 obj/forward_b.o \
 
#% cpp compile
$(BUILDDIR)/%.$(OBJEXT): $(SOLVERSRC)/*/%.$(CEXT)
	$(CC) $(CFLAGS) $(MOD) $(INC) -c -o $@ $<

#% f77 compile
$(BUILDDIR)/%.$(OBJEXT): $(SOLVERSRC)/*/%.$(F77EXT)
	$(FC) $(F77FLAGS) $(MOD) $(INC) -c -o $@ $<

#% f90 compile
$(BUILDDIR)/%.$(OBJEXT): $(SOLVERSRC)/*/%.$(F90EXT)
	$(FC) $(F90FLAGS) $(MOD) $(INC) -c -o $@ $<
 
#% Making wrappers (f90wrap)
wrappers:
	@echo "********************************************"
	@echo ""
	@echo " Making wrappers "
	@echo ""
	@echo "********************************************"
	f90wrap -m $(SHAREDLIB) $(SOLVERMODWRAP) -k kind_map --package
	
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
	cd $(MESHSRC) ; python3 -m numpy.f2py -c -m _meshing meshing.f90 skip: mask_upstream_cells downstream_cell_drained_area
	
#% Finalize compilation with mv, rm and sed
finalize:
	mv f90wrap_* $(SOLVERSRC)/f90wrap/.
	mv $(SHAREDLIB)/mw_* $(SOLVERSRC)/.
	mv _$(SHAREDLIB)* $(SOLVERSRC)/.
	rm -rf $(SHAREDLIB)
	bash sed_f90wrap.sh
	
#% Generating tapenade files (adjoint and tangent linear models)
tap:
	cd $(TAPENADEDIR) ; make
	
#% Clean
clean:
	@$(RM) -rf $(EXTDIR)
	@$(RM) -rf $(BUILDDIR)
	@$(RM) -rf src.*
	@$(RM) -rf *egg-info
	@$(RM) -rf $(SOLVERSRC)/mw_*
	@$(RM) -rf $(SOLVERSRC)/_$(SHAREDLIB)*
	@$(RM) -rf $(SOLVERSRC)/f90wrap
	@$(RM) -rf $(MESHSRC)/*.so
