FILES=m_run.f90 m_setup.f90 m_mesh.f90 m_input_data.f90 m_parameters.f90 m_states.f90 m_output.f90 m_common.f90

TAPENADE=./tapenade_3.16/

all: b

b: $(FILES)
	$(TAPENADE)/bin/tapenade -backward -fixinterface -msglevel 100 -o ADJ -head m_run.run_direct \
	-vars "parameters" \
	-outvars "cost" \
	-O $(PWD)/backward $(FILES)
