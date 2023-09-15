SRC := ../smash/fcore
FILES := mwd_*.f90 md_*.f90 forward.f90
SRC_FILES := $(addprefix $(SRC)/*/,$(FILES))
ADJ_FILE = forward_db.f90
TAPENADE := ./tapenade_3.16

all: pre-sed bd post-sed mv clean

tap-cmp: pre-sed bd post-sed cmp clean

pre-sed:
	# pre-sed
	# Remove lines between !$$AD start-exclude and !$$AD end-exclude
	cp $(SRC_FILES) .
	for file in $(FILES); do \
	    sed -i '/!\$$AD[[:space:]]*start-exclude/,/!\$$AD[[:space:]]*end-exclude/d' "$$file"; \
	done

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

post-sed:
	# sed
	# Tapenade creates new derived type when not all values inside the derived type are used inside the graph.
	# Therefore, just seding the two derived type affected.
	sed -i 's/TYPE(PARAMETERSDT_DIFF)/TYPE(PARAMETERSDT)/g' $(ADJ_FILE)
	sed -i 's/TYPE(OUTPUTDT_DIFF)/TYPE(OUTPUTDT)/g' $(ADJ_FILE)
mv:
	# mv
	mv $(ADJ_FILE) $(SRC)/forward/.

cmp:
	# compare old and new tap file
	cmp $(ADJ_FILE) $(SRC)/forward/$(ADJ_FILE)

clean:
	rm -f *.f90

.PHONY: all tap-cmp bd pre-sed post-sed mv cmp clean
