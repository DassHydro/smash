#!/bin/bash

# sed wrapped and differentiated DT
sed_mwd_DT () {

	# Add copy to class method
	sed -i "/.*join(ret).*/a \\\n\tdef copy(self):\n\t\treturn "$1"_copy(self)" ./smash/solver/_mwd_$1.py
	
	# Check for finalise method (depend on which machine/compiler its used)
	sed -i "/.*_finalise/i \\\t\t\ttry:" ./smash/solver/_mwd_$1.py
	sed -i "/.*finalise/s/^/\t/" ./smash/solver/_mwd_$1.py
	sed -i "/.*_finalise/a \\\t\t\texcept:\n\t\t\t\tpass" ./smash/solver/_mwd_$1.py

}


mwd_DT=("setup" "mesh" "input_data" "parameters" "states" "output")

for dt in ${mwd_DT[@]}; do
  
  sed_mwd_DT $dt
  
done

decode=".tobytes(order='F').decode('utf-8').split()"

# Apply decode to mdw_common chararray (avoid tobytes(order='F').decode('utf-8').split())
	## _mwd_common.py
sed -i "/import logging/a \import numpy" ./smash/solver/_mwd_common.py
sed -i "/return name_parameters/i \\\tname_parameters = numpy.array(name_parameters$decode)" ./smash/solver/_mwd_common.py
sed -i "/return name_states/i \\\tname_states = numpy.array(name_states$decode)" ./smash/solver/_mwd_common.py

# Change relative to absolute _solver import
sed -i "0,/import _solver/s//from smash.solver import _solver/" ./smash/solver/_mw*.py

# Replace tab with 4 spaces
sed -i 's/\t/    /g' ./smash/solver/_mw*.py
