FC := gfortran
CC := gcc
BUILDDIR := obj
TARGET := SMASH

#FFLAGS := -cpp -O3 -march=native -funroll-loops -ffast-math -fPIC -fopenmp
F90FLAGS := -Wall -Wextra -fPIC -fmax-errors=1 -fopenmp -cpp -g -fcheck=all -fbacktrace -fcheck-array-temporaries
F77FLAGS := -O3 -march=native -funroll-loops -ffast-math -fPIC
CFLAGS := -g -O3 -march=native -ffast-math -fPIC

F90EXT := f90
F77EXT := f
CEXT := c

OBJEXT := o

SMASHDIR := smash
TAPENADEDIR := tapenade
SOLVERSRC := smash/solver
MESHSRC := smash/mesh

INC := -I$(BUILDDIR)
MOD := -J$(BUILDDIR)

SHAREDLIB := solver
SOLVERMODWRAP := $(SOLVERSRC)/module/mw_*.f90
OBJWRAP := $(BUILDDIR)/*.o
SOLVERWRAPPERS := f90wrap*.f90

all: directories cpp f77 f90 wrappers module meshing finalize

directories:
	@echo "********************************************"
	@echo ""
	@echo " Making directories "
	@echo ""
	@echo "********************************************"
	@mkdir -p $(BUILDDIR)
	@mkdir -p $(SOLVERSRC)/f90wrap/
	
	
cpp: \
 obj/adStack.o
 
 
f77: \
 obj/adBuffer.o
 
 
f90: \
 obj/adStack.o \
 obj/adBuffer.o \
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
 obj/mw_optimize.o \
 obj/mw_utils.o \
 obj/forward.o \
 obj/forward_b.o \
 obj/forward_d.o \
 
$(BUILDDIR)/%.$(OBJEXT): $(SOLVERSRC)/*/%.$(CEXT)
	$(CC) $(CFLAGS) $(MOD) $(INC) -c -o $@ $<
	
$(BUILDDIR)/%.$(OBJEXT): $(SOLVERSRC)/*/%.$(F77EXT)
	$(FC) $(F77FLAGS) $(MOD) $(INC) -c -o $@ $<

$(BUILDDIR)/%.$(OBJEXT): $(SOLVERSRC)/*/%.$(F90EXT)
	$(FC) $(F90FLAGS) $(MOD) $(INC) -c -o $@ $<
 
 
wrappers:
	@echo "********************************************"
	@echo ""
	@echo " Making wrappers "
	@echo ""
	@echo "********************************************"
	f90wrap -m $(SHAREDLIB) $(SOLVERMODWRAP) -k kind_map --package
	
	
module:
	@echo "********************************************"
	@echo ""
	@echo " Making module extension "
	@echo ""
	@echo "********************************************"
	f2py-f90wrap -c --fcompiler=gfortran --f90flags='-cpp -fopenmp -fPIC -fmax-errors=1 -Iobj -Jobj' -lgomp --arch='-march=native' --opt='-O3 -funroll-loops -ffast-math' --build-dir . -m _$(SHAREDLIB) $(OBJWRAP) $(SOLVERWRAPPERS)


meshing:
	cd $(MESHSRC) ; python3 -m numpy.f2py -c -m _meshing _meshing.f90 skip: mask_upstream_cells downstream_cell_drained_area
	

finalize:
	mv f90wrap_* $(SOLVERSRC)/f90wrap/.
	mv $(SHAREDLIB)/mw_* $(SOLVERSRC)/.
	mv _$(SHAREDLIB)* $(SOLVERSRC)/.
	rm -rf $(SHAREDLIB)
	bash sed_f90wrap.sh
	
	
tap:
	cd $(TAPENADEDIR) ; make


clean:
	@$(RM) -rf $(EXTDIR)
	@$(RM) -rf $(BUILDDIR)
	@$(RM) -rf src.*
	@$(RM) -rf *egg-info
	@$(RM) -rf $(SOLVERSRC)/mw_*
	@$(RM) -rf $(SOLVERSRC)/_$(SHAREDLIB)*
	@$(RM) -rf $(SOLVERSRC)/f90wrap
