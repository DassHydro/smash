SRC := ../smash/solver
NAME := mwd_*.f90 md_*.f90 forward.f90
FILES := $(addprefix $(SRC)/*/,$(NAME))
TAPENADE := ./tapenade_3.16

all: bd mv

tap-cmp: bd cmp

bd:
	$(TAPENADE)/bin/tapenade \
	-b \
	-d \
	-fixinterface \
	-noisize \
	-openmp \
	-context \
	-msglevel 100 \
	-adjvarname %_b \
	-tgtvarname %_d \
	-o forward \
	-head "base_forward_run(parameters.control.x)\(output)" \
	-O $(PWD) $(FILES)

mv:
	# mv
	mv forward_db.f90 $(SRC)/forward/.

cmp:
	# compare old and new tap file
	cmp forward_db.f90 $(SRC)/forward/forward_db.f90

.PHONY: all tap-cmp bd mv cmp

