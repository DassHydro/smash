#!/bin/bash

sed_DT () {
	
	# Add copy method
	sed -i "/import logging/a \from m_utils import "$1"_derived_type_copy" ./smash/solver/m_$1.py
	sed -i "/.*join(ret).*/a \\\n\tdef copy(self):\n\t\treturn "$1"_derived_type_copy(self)" ./smash/solver/m_$1.py
	
	# Check for finalise method (depend on which machine/compiler its used)
	sed -i "/.*_finalise/i \\\t\t\ttry:" ./smash/solver/m_$1.py
	sed -i "/.*finalise/s/^/\t/" ./smash/solver/m_$1.py
	sed -i "/.*_finalise/a \\\t\t\texcept:\n\t\t\t\tpass" ./smash/solver/m_$1.py
	
	# Replace tab with 4 spaces
	sed -i 's/\t/    /g' ./smash/solver/m_$1.py
}

derived_type=("setup" "mesh" "input_data" "parameters" "states")

for dt in ${derived_type[@]}; do
  
  sed_DT $dt
  
done
