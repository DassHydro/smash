#!/bin/bash

# SMASH Adjoint generation

SRC=../smash/solver/

mkdir backward/
mkdir forward/

cp $SRC/wrapped_module/mw_run.f90 $SRC/wrapped_module/mw_setup.f90 $SRC/wrapped_module/mw_mesh.f90 $SRC/wrapped_module/mw_input_data.f90 $SRC/wrapped_module/mw_parameters.f90 $SRC/wrapped_module/mw_states.f90 $SRC/wrapped_module/mw_output.f90 $SRC/module/m_common.f90 .

make all

rm mw_run.f90 mw_setup.f90 mw_mesh.f90 mw_input_data.f90 mw_parameters.f90 mw_states.f90 mw_output.f90 m_common.f90
