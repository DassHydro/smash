SRC := ../smash/fcore
NAME := mwd_*.f90 md_*.f90 forward.f90
FILES := $(addprefix $(SRC)/*/,$(NAME))
TAPENADE := ./tapenade_3.16

all: bd sed mv

tap-cmp: bd sed cmp

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
	-head "base_forward_run(parameters.control.x)\(output.cost)" \
	-O $(PWD) $(FILES)

sed:
	# sed. Tapenade creates new derived type when not all values inside the derived type are used inside the graph.
	# Therefore, just seding the two derived type affected.
	sed -i 's/TYPE(PARAMETERSDT_DIFF)/TYPE(PARAMETERSDT)/g' forward_db.f90
	sed -i 's/TYPE(OUTPUTDT_DIFF)/TYPE(OUTPUTDT)/g' forward_db.f90
mv:
	# mv
	mv forward_db.f90 $(SRC)/forward/.

cmp:
	# compare old and new tap file
	cmp forward_db.f90 $(SRC)/forward/forward_db.f90

.PHONY: all tap-cmp bd sed mv cmp

