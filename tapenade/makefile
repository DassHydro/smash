SRC=../smash/solver
NAME=forward.f90 mw_setup.f90 mw_mesh.f90 mw_input_data.f90 mw_parameters.f90 mw_states.f90 mw_output.f90 m_common.f90 m_operator.f90 mw_cost.f90
FILES=$(addprefix $(SRC)/*/,$(NAME))
TAPENADE=./tapenade_3.16/

all: b d finalize

b: 
	$(TAPENADE)/bin/tapenade -b -openmp -context -fixinterface -msglevel 100 -adjvarname %_b -o forward -head "forward(parameters, states, cost)\(cost)" -O $(PWD) $(FILES)
	
d:
	$(TAPENADE)/bin/tapenade -d -openmp -context -fixinterface -msglevel 100 -tgtvarname %_d -o forward -head "forward(parameters, states, cost)\(cost)" -O $(PWD) $(FILES)

finalize:
	sed -i "/.*DIFF/ s/DIFF/DIFF_B/g" forward_b.f90
	sed -i 's/.*CALL.*COMPUTE\_JOBS\_B.*/  CALL COMPUTE\_JOBS(setup, mesh, input_data, output, cost)\n&/' forward_b.f90
	sed -i "/.*DIFF/ s/DIFF/DIFF_D/g" forward_d.f90
	mv forward_*.f90 $(SRC)/forward/.
