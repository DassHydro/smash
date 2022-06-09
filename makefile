FC := gfortran
TARGETDIR := bin
BUILDDIR := obj
TARGET := SMASH

F90FLAGS := -cpp -O3 -march=native -funroll-loops -ffast-math -fPIC
F90SRC := smash/f90
F90EXT := f90
OBJEXT := o
PYSRC := smash

INC := -I$(BUILDDIR)
MOD := -J$(BUILDDIR)

SHAREDLIB := wrapping
F90MODWRAP := $(F90SRC)/wrapped_module/*.f90
OBJWRAP := $(BUILDDIR)/*.o
F90WRAPPERS := $(F90SRC)/$(SHAREDLIB)/*.f90

all: directories $(TARGET) wrappers module

directories:
	@echo "********************************************"
	@echo ""
	@echo " Making directories "
	@echo ""
	@echo "********************************************"
	@mkdir -p $(TARGETDIR)
	@mkdir -p $(BUILDDIR)
	@mkdir -p $(F90SRC)/$(SHAREDLIB)
	@mkdir -p $(PYSRC)/$(SHAREDLIB)

wrappers:
	@echo "********************************************"
	@echo ""
	@echo " Making wrappers "
	@echo ""
	@echo "********************************************"
	f90wrap -m $(SHAREDLIB) $(F90MODWRAP) -k kind_map --package
	mv f90wrap_*.f90 $(F90SRC)/$(SHAREDLIB)/.
	
module:
	@echo "********************************************"
	@echo ""
	@echo " Making module extension "
	@echo ""
	@echo "********************************************"
	f2py-f90wrap -c --fcompiler=gfortran --f90flags='-cpp -fPIC -fmax-errors=1 -Iobj -Jobj' --arch='-march=native' --opt='-O3 -funroll-loops -ffast-math' --build-dir . -m _$(SHAREDLIB) $(OBJWRAP) $(F90WRAPPERS)
	mv $(SHAREDLIB)/m_* $(PYSRC)/$(SHAREDLIB)/.
	mv _$(SHAREDLIB)* $(PYSRC)/$(SHAREDLIB)/.
	rm -rf $(SHAREDLIB)

clean:
	@$(RM) -rf $(TARGETDIR)
	@$(RM) -rf $(EXTDIR)
	@$(RM) -rf $(BUILDDIR)
	@$(RM) -rf src.*
	@$(RM) -rf *egg-info

$(TARGET): \
 obj/m_common.o \
 obj/m_setup.o \
 obj/m_mesh.o \
 $(F90SRC)/main.f90
	$(FC) $(F90FLAGS) $(MOD) $(INC) -o $(TARGETDIR)/$(TARGET) $^

$(BUILDDIR)/%.$(OBJEXT): $(F90SRC)/*/%.$(F90EXT)
	@mkdir -p $(dir $@)
	$(FC) $(F90FLAGS) $(MOD) $(INC) -c -o $@ $<
