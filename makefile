FC := gfortran
BUILDDIR := obj
TARGET := SMASH

#FFLAGS := -cpp -O3 -march=native -funroll-loops -ffast-math -fPIC -fopenmp
FFLAGS := -Wall -Wextra -fPIC -fmax-errors=1 -fopenmp -cpp -g -fcheck=all -fbacktrace -fcheck-array-temporaries
SOLVERSRC := smash/solver
MESHSRC := smash/mesh
FEXT := f90
OBJEXT := o
SMASHDIR := smash

INC := -I$(BUILDDIR)
MOD := -J$(BUILDDIR)

SHAREDLIB := solver
SOLVERMODWRAP := $(SOLVERSRC)/module/mw_*.f90
OBJWRAP := $(BUILDDIR)/*.o
SOLVERWRAPPERS := $(SOLVERSRC)/*.f90

all: directories $(TARGET) wrappers module

directories:
	@echo "********************************************"
	@echo ""
	@echo " Making directories "
	@echo ""
	@echo "********************************************"
	@mkdir -p $(BUILDDIR)

wrappers:
	@echo "********************************************"
	@echo ""
	@echo " Making wrappers "
	@echo ""
	@echo "********************************************"
	f90wrap -m $(SHAREDLIB) $(SOLVERMODWRAP) -k kind_map --package
	mv f90wrap_*.f90 $(SOLVERSRC)/.
	
module:
	@echo "********************************************"
	@echo ""
	@echo " Making module extension "
	@echo ""
	@echo "********************************************"
	f2py-f90wrap -c --fcompiler=gfortran --f90flags='-cpp -fopenmp -fPIC -fmax-errors=1 -Iobj -Jobj' -lgomp --arch='-march=native' --opt='-O3 -funroll-loops -ffast-math' --build-dir . -m _$(SHAREDLIB) $(OBJWRAP) $(SOLVERWRAPPERS)
	mv $(SHAREDLIB)/mw_* $(SOLVERSRC)/.
	mv _$(SHAREDLIB)* $(SOLVERSRC)/.
	rm -rf $(SHAREDLIB)
	cd $(MESHSRC) ; python3 -m numpy.f2py -c -m _meshing _meshing.f90 skip: mask_upstream_cells downstream_cell_drained_area
	bash sed_f90wrap.sh

clean:
	@$(RM) -rf $(EXTDIR)
	@$(RM) -rf $(BUILDDIR)
	@$(RM) -rf src.*
	@$(RM) -rf *egg-info
	@$(RM) -rf $(SOLVERSRC)/mw_*
	@$(RM) -rf $(SOLVERSRC)/_$(SHAREDLIB)*
	@$(RM) -rf $(SOLVERSRC)/*f90
	@$(RM) -rf $(SOLVERSRC)/*o


$(TARGET): \
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
 
$(BUILDDIR)/%.$(OBJEXT): $(SOLVERSRC)/*/%.$(FEXT)
	@mkdir -p $(dir $@)
	$(FC) $(FFLAGS) $(MOD) $(INC) -c -o $@ $<
