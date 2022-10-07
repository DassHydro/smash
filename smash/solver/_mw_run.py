"""
Module mw_run


Defined at smash/solver/module/mw_run.f90 lines 9-120

This module `mw_run` encapsulates all SMASH run.
This module is wrapped.

contains

[1] forward_run
[2] adjoint_run
[3] tangent_linear_run
"""
from __future__ import print_function, absolute_import, division
from smash.solver import _solver
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

def forward_run(self, mesh, input_data, parameters, parameters_bgd, states, \
    states_bgd, output, verbose):
    """
    forward_run(self, mesh, input_data, parameters, parameters_bgd, states, \
        states_bgd, output, verbose)
    
    
    Defined at smash/solver/module/mw_run.f90 lines 21-44
    
    Parameters
    ----------
    setup : Setupdt
    mesh : Meshdt
    input_data : Input_Datadt
    parameters : Parametersdt
    parameters_bgd : Parametersdt
    states : Statesdt
    states_bgd : Statesdt
    output : Outputdt
    verbose : bool
    
    Notes
    -----
    
    Forward run subroutine
    Given SetupDT, MeshDT, Input_DataDT, ParametersDT,
    ParametersDT_bgd, StatesDT, StatesDT_bgd, OutputDT and logical verbose value,
    it computes a forward run. Verbose argument allow to print:
    "</> Forward Model M(k)"
    
    Calling forward from forward/forward.f90  Y = M(k)
    """
    _solver.f90wrap_forward_run(setup=self._handle, mesh=mesh._handle, \
        input_data=input_data._handle, parameters=parameters._handle, \
        parameters_bgd=parameters_bgd._handle, states=states._handle, \
        states_bgd=states_bgd._handle, output=output._handle, verbose=verbose)

def adjoint_run(self, mesh, input_data, parameters, states, output):
    """
    adjoint_run(self, mesh, input_data, parameters, states, output)
    
    
    Defined at smash/solver/module/mw_run.f90 lines 47-84
    
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
    
    Forward run subroutine
    Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
    it computes an adjoint run.
    
    Calling forward_b from forward/forward_b.f90 dk* = (dM/dk)* (k) . dY*
    WIP
    """
    _solver.f90wrap_adjoint_run(setup=self._handle, mesh=mesh._handle, \
        input_data=input_data._handle, parameters=parameters._handle, \
        states=states._handle, output=output._handle)

def tangent_linear_run(self, mesh, input_data, parameters, states, output):
    """
    tangent_linear_run(self, mesh, input_data, parameters, states, output)
    
    
    Defined at smash/solver/module/mw_run.f90 lines 87-120
    
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
    
    Forward run subroutine
    Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
    it computes an tangent linear run.
    
    Calling forward_d from forward/forward_d.f90 dY = (dM/dk) (k) . dk
    WIP
    """
    _solver.f90wrap_tangent_linear_run(setup=self._handle, mesh=mesh._handle, \
        input_data=input_data._handle, parameters=parameters._handle, \
        states=states._handle, output=output._handle)


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module "mw_run".')

for func in _dt_array_initialisers:
    func()
