"""
Module mw_optimize


Defined at smash/solver/module/mw_optimize.f90 lines 15-558

This module `mw_optimize` encapsulates all SMASH optimize.
This module is wrapped.

contains

[1] optimize_sbs
[2] transformation
[3] inv_transformation
[4] optimize_lbfgsb
[5] optimize_matrix_to_vector
[6] optimize_vector_to_matrix
[7] normalize_matrix
[8] unnormalize_matrix
[9] optimize_message
"""
from __future__ import print_function, absolute_import, division
from smash.solver import _solver
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

def optimize_sbs(self, mesh, input_data, parameters, states, output):
    """
    optimize_sbs(self, mesh, input_data, parameters, states, output)
    
    
    Defined at smash/solver/module/mw_optimize.f90 lines 32-283
    
    Parameters
    ----------
    setup : Setupdt
    mesh : Meshdt
    input_data : Input_Datadt
    parameters : Parametersdt
    states : Statesdt
    output : Outputdt
    
    Notes
    -----
    
    Step By Step optimization subroutine
    
    Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
    it returns the result of a step by step optimization.
    argmin(theta) = J(theta)
    
    Calling forward from forward/forward.f90  Y  = M(k)
    =========================================================================================================== \
        %
    Initialisation
    =========================================================================================================== \
        %
    ======================================================================================================= \
        %
    Optimize
    ======================================================================================================= \
        %
    ======================================================================================================= \
        %
    Iterate writting
    ======================================================================================================= \
        %
    ======================================================================================================= \
        %
    Convergence DDX < 0.01
    ======================================================================================================= \
        %
    ======================================================================================================= \
        %
    Maximum Number of Iteration
    ======================================================================================================= \
        %
    """
    _solver.f90wrap_optimize_sbs(setup=self._handle, mesh=mesh._handle, \
        input_data=input_data._handle, parameters=parameters._handle, \
        states=states._handle, output=output._handle)

def optimize_lbfgsb(self, mesh, input_data, parameters, states, output):
    """
    optimize_lbfgsb(self, mesh, input_data, parameters, states, output)
    
    
    Defined at smash/solver/module/mw_optimize.f90 lines 317-437
    
    Parameters
    ----------
    setup : Setupdt
    mesh : Meshdt
    input_data : Input_Datadt
    parameters : Parametersdt
    states : Statesdt
    output : Outputdt
    
    Notes
    -----
    
    L-BFGS-B optimization subroutine
    
    Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
    it returns the result of a l-bfgs-b optimization.
    argmin(theta) = J(theta)
    
    Calling forward_b from forward/forward_b.f90  dk* = (dM/dk)* (k) . dY*
    Calling setulb from optimize/lbfgsb.f
    """
    _solver.f90wrap_optimize_lbfgsb(setup=self._handle, mesh=mesh._handle, \
        input_data=input_data._handle, parameters=parameters._handle, \
        states=states._handle, output=output._handle)


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module \
        "mw_optimize".')

for func in _dt_array_initialisers:
    func()
