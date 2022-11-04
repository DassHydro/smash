SRC=../smash/solver
NAME=md_common.f90 mwd_setup.f90 mwd_mesh.f90 mwd_input_data.f90 mwd_parameters.f90 mwd_states.f90 mwd_output.f90 md_gr_operator.f90 md_routing_operator.f90 mwd_parameters_manipulation.f90 mwd_states_manipulation.f90 mwd_cost.f90 forward.f90 
FILES=$(addprefix $(SRC)/*/,$(NAME))
TAPENADE=./tapenade_3.16/

all: bd finalize

bd:
	$(TAPENADE)/bin/tapenade \
	-b \
	-d \
	-fixinterface \
	-msglevel 100 \
	-adjvarname %_b \
	-tgtvarname %_d \
	-o forward \
	-head "base_forward(parameters, states)\(cost)" \
	-head "base_hyper_forward(hyper_parameters, hyper_states)\(cost)" \
	-O $(PWD) $(FILES)


finalize:
	# sed forward_b
	sed -i "s/.*CALL.COMPUTE\_COST\_B.*/  CALL SET\_PARAMETERS(parameters\_b, 0.0_4)\n  CALL SET\_STATES(states\_b, 0.0_4)\n&/" forward_db.f90
	
	# sed hyper_forward_b
	sed -i "s/.*CALL.HYPER\_COMPUTE\_COST\_B.*/  CALL SET\_HYPER\_PARAMETERS(hyper\_parameters\_b, 0.0_4)\n  CALL SET\_HYPER\_STATES(hyper\_states\_b, 0.0_4)\n&/" forward_db.f90
	
	# sed \t for 4 spaces
	sed -i "s/\t/    /g" forward_db.f90
	
	# mv
	mv forward_db.f90 $(SRC)/forward/.
	
