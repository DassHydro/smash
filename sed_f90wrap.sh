#!/bin/bash

sed_copy () {

	sed -i "/import logging/a \from m_utils import "$1"_derived_type_copy" ./smash/solver/m_$1.py
	sed -i "/.*join(ret).*/a \\\n\tdef copy(self):\n\t\treturn "$1"_derived_type_copy(self)" ./smash/solver/m_$1.py

	sed -i 's/\t/    /g' ./smash/solver/m_$1.py
}

derived_type=("setup" "mesh" "input_data")

for dt in ${derived_type[@]}; do
  
  sed_copy $dt
  
done
