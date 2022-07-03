SRC=../smash/solver
NAME=forward.f90 mwd_setup.f90 mwd_mesh.f90 mwd_input_data.f90 mwd_parameters.f90 mwd_states.f90 mwd_output.f90 mwd_common.f90 md_operator.f90 mwd_cost.f90
FILES=$(addprefix $(SRC)/*/,$(NAME))
TAPENADE=./tapenade_3.16/

all: b d finalize

b: 
	$(TAPENADE)/bin/tapenade -b -openmp -context -fixinterface -msglevel 100 -adjvarname %_b -o forward -head "forward(parameters, states, cost)\(cost)" -O $(PWD) $(FILES)
	
d:
	$(TAPENADE)/bin/tapenade -d -openmp -context -fixinterface -msglevel 100 -tgtvarname %_d -o forward -head "forward(parameters, states, cost)\(cost)" -O $(PWD) $(FILES)

finalize:
	# sed forward_b
	sed -i "/.*DIFF/ s/DIFF/DIFF_B/g" forward_b.f90
	sed -i "s/.*CALL.*COMPUTE\_JOBS\_B.*/  CALL COMPUTE\_JOBS(setup, mesh, input_data, output, cost)\n&/" forward_b.f90
	#sed -i "/output\_b%qsim\_domain = 0.0\_.*/s/^/  IF (.not. setup%sparse_storage)/" forward_b.f90
	#sed -i "/output\_b%sparse\_qsim\_domain = 0.0\_.*/s/^/  IF (setup%sparse_storage)/" forward_b.f90
	
	# sed forward_b 
	sed -i "/.*DIFF/ s/DIFF/DIFF_D/g" forward_d.f90
	#sed -i "/output\_d%qsim\_domain = 0.0\_.*/s/^/  IF (.not. setup%sparse_storage)/" forward_d.f90
	#sed -i "/output\_d%sparse\_qsim\_domain = 0.0\_.*/s/^/  IF (setup%sparse_storage)/" forward_d.f90
	
	# mv
	mv forward_*.f90 $(SRC)/forward/.
	
