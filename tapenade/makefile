SRC=../smash/solver
NAME=forward.f90 mwd_setup.f90 mwd_mesh.f90 mwd_input_data.f90 mwd_parameters.f90 mwd_states.f90 mwd_output.f90 mwd_common.f90 md_routine.f90 md_operator.f90 mwd_cost.f90
FILES=$(addprefix $(SRC)/*/,$(NAME))
TAPENADE=./tapenade_3.16/

all: b d finalize

b: 
	$(TAPENADE)/bin/tapenade -b -fixinterface -msglevel 100 -adjvarname %_b -o forward -head "forward(parameters, states, cost)\(cost)" -O $(PWD) $(FILES)
	
d:
	$(TAPENADE)/bin/tapenade -d -fixinterface -msglevel 100 -tgtvarname %_d -o forward -head "forward(parameters, states, cost)\(cost)" -O $(PWD) $(FILES)

finalize:
	# sed forward_b
	sed -i "/.*DIFF/ s/DIFF/DIFF_B/g" forward_b.f90
	sed -i "s/.*CALL.*COMPUTE\_COST\_B.*/  CALL COMPUTE\_COST(setup, mesh, input_data, parameters, parameters_bgd, states_imd, states_bgd, output, cost)\n&/" forward_b.f90
	sed -i "s/.*CALL.*COMPUTE\_COST\_B.*/  CALL SET0_PARAMETERS(parameters\_b)\n  CALL SET0_STATES(states\_b)\n  CALL SET0_STATES(states\_imd\_b)\n&/" forward_b.f90
	
	# sed forward_d 
	sed -i "/.*DIFF/ s/DIFF/DIFF_D/g" forward_d.f90
	
	# mv
	mv forward_*.f90 $(SRC)/forward/.
	
