#!/bin/bash

# SMASH Adjoint generation

SRC=../smash/solver/

mkdir backward/
mkdir forward/

cp $SRC/wrapped_module/m_run.f90 $SRC/wrapped_module/m_setup.f90 $SRC/wrapped_module/m_mesh.f90 $SRC/wrapped_module/m_input_data.f90 $SRC/wrapped_module/m_parameters.f90 $SRC/wrapped_module/m_states.f90 $SRC/wrapped_module/m_output.f90 $SRC/module/m_common.f90 .

make all

rm m_run.f90 m_setup.f90 m_mesh.f90 m_input_data.f90 m_parameters.f90 m_states.f90 m_output.f90 m_common.f90
