#!/bin/bash

# sed wrapped and differentiated DT
sed_DT () {

	# Add copy to class method
	sed -i "/from __future__/a \from smash.solver._mw_routine import copy_$1" ./smash/solver/_*$1.py
	sed -i "/.*join(ret).*/a \\\n\tdef copy(self):\n\t\treturn copy_$1(self)" ./smash/solver/_*$1.py
	
	# Check for finalise method (depend on which machine/compiler its used)
	sed -i "/.*_finalise/i \\\t\t\ttry:" ./smash/solver/_*$1.py
	sed -i "/.*finalise/s/^/\t/" ./smash/solver/_*$1.py
	sed -i "/.*_finalise/a \\\t\t\texcept:\n\t\t\t\tpass" ./smash/solver/_*$1.py

}


main_DT=("setup" "mesh" "input_data" "parameters" "states" "output")

for dt in ${main_DT[@]}; do
  
  sed_DT $dt
  
done

# private f90wrap module attributes
priv_attr=(
"mwd_setup.ntime_step"
"mwd_setup.nd"
"mwd_setup.algorithm"
"mwd_setup.jobs_fun"
"mwd_setup.mapping"
"mwd_setup.jreg_fun"
"mwd_setup.wjreg"
"mwd_setup.optim_start_step"
"mwd_setup.optim_parameters"
"mwd_setup.optim_states"
"mwd_setup.lb_parameters"
"mwd_setup.ub_parameters"
"mwd_setup.lb_states"
"mwd_setup.ub_states"
"mwd_setup.maxiter"
"mwd_mesh.wgauge"
"mwd_mesh.rowcol_to_ind_sparse"
"mwd_mesh.local_active_cell"
)

for args in ${priv_attr[@]}; do
  
  IFS=. read -r f attr <<< $args
  sed -i "s/def ${attr}/def _${attr}/g" ./smash/solver/*${f}*.py
  sed -i "s/@${attr}/@_${attr}/g" ./smash/solver/*${f}*.py
  sed -i "s/self.${attr}/self._${attr}/g" ./smash/solver/*${f}*.py
  sed -i "/ret.append.*${attr}.*/d" ./smash/solver/*${f}*.py
  
done

decode=".tobytes(order='F').decode('utf-8').split()"

# Apply decode to mdw_common chararray (avoid tobytes(order='F').decode('utf-8').split())
	## _mwd_common.py
sed -i "/import logging/a \import numpy" ./smash/solver/_mwd_common.py
sed -i "/return name_parameters/i \\\tname_parameters = numpy.array(name_parameters$decode)" ./smash/solver/_mwd_common.py
sed -i "/return name_states/i \\\tname_states = numpy.array(name_states$decode)" ./smash/solver/_mwd_common.py

# Change relative to absolute solver/_solver
sed -i "0,/import _solver/s//from smash.solver import _solver/" ./smash/solver/_mw*.py
sed -i "s/from solver/from smash.solver/g" ./smash/solver/_mw*.py

# Replace tab with 4 spaces
sed -i "s/\t/    /g" ./smash/solver/_mw*.py
