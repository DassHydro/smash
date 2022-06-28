FILES=mw_run.f90 mw_setup.f90 mw_mesh.f90 mw_input_data.f90 mw_parameters.f90 mw_states.f90 mw_output.f90 m_common.f90 m_operator.f90 mw_cost.f90

TAPENADE=./tapenade_3.16/

all: b

b: $(FILES)
	$(TAPENADE)/bin/tapenade -backward -openmp -context -fixinterface -msglevel 100 -o ADJ -head "mw_run.direct_model(parameters, states, cost)\(cost)" \
	-O $(PWD)/backward $(FILES)
