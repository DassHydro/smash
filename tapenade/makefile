SRC := ../smash/solver
NAME := mwd_*.f90 md_*.f90 forward.f90
FILES := $(addprefix $(SRC)/*/,$(NAME))
TAPENADE := ./tapenade_3.16/

all: bd sed mv

tap-cmp: bd sed cmp

bd:
	$(TAPENADE)/bin/tapenade \
	-b \
	-d \
	-fixinterface \
	-noisize \
	-msglevel 100 \
	-adjvarname %_b \
	-tgtvarname %_d \
	-o forward \
	-head "base_forward(parameters, states)\(cost)" \
	-head "base_hyper_forward(hyper_parameters, hyper_states)\(cost)" \
	-O $(PWD) $(FILES)

sed:
	# sed forward_b
	sed -i "s/.*CALL.COMPUTE\_COST\_B.*/  CALL SET\_PARAMETERS(mesh, parameters\_b, 0.0_4)\n  CALL SET\_STATES(mesh, states\_b, 0.0_4)\n&/" forward_db.f90
	
	# sed hyper_forward_b
	sed -i "s/.*CALL.HYPER\_COMPUTE\_COST\_B.*/  CALL SET\_HYPER\_PARAMETERS(setup, hyper\_parameters\_b, 0.0_4)\n  CALL SET\_HYPER\_STATES(setup, hyper\_states\_b, 0.0_4)\n&/" forward_db.f90
	
	# sed \t for 4 spaces
	sed -i "s/\t/    /g" forward_db.f90

mv:
	# mv
	mv forward_db.f90 $(SRC)/forward/.

cmp:
	# compare old and new tap file
	cmp forward_db.f90 $(SRC)/forward/forward_db.f90

.PHONY: all tap-cmp bd sed mv cmp

