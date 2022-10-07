"""
Module mw_adjoint_test


Defined at smash/solver/module/mw_adjoint_test.f90 lines 8-141

This module `mw_adjoint_test` encapsulates all SMASH adjoint_test.
This module is wrapped.

contains

[1] scalar_product_test
[2] gradient_test
"""
from __future__ import print_function, absolute_import, division
from smash.solver import _solver
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

def scalar_product_test(self, mesh, input_data, parameters, states, output):
    """
    scalar_product_test(self, mesh, input_data, parameters, states, output)
    
    
    Defined at smash/solver/module/mw_adjoint_test.f90 lines 22-79
    
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
    
    Scalar Product Test subroutine
    
    Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
    it returns the results of scalar product test.
    sp1 = <dY*, dY>
    sp2 = <dk*, dk>
    
    Calling forward_d from forward/forward_d.f90  dY  = (dM/dk) (k) . dk
    Calling forward_b from forward/forward_b.f90  dk* = (dM/dk)* (k) . dY*
    cost_b reset at the end of forward_b ...
    No perturbation has been set to states_d
    """
    _solver.f90wrap_scalar_product_test(setup=self._handle, mesh=mesh._handle, \
        input_data=input_data._handle, parameters=parameters._handle, \
        states=states._handle, output=output._handle)

def gradient_test(self, mesh, input_data, parameters, states, output):
    """
    gradient_test(self, mesh, input_data, parameters, states, output)
    
    
    Defined at smash/solver/module/mw_adjoint_test.f90 lines 81-141
    
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
    
    Gradient Test subroutine
    
    Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
    it returns the results of gradient test.
    Ia = (Yadk - Y) / (a dk* . dk)
    
    Calling forward from forward/forward.f90      Y   = M(k)
    Calling forward_b from forward/forward_b.f90  dk* = (dM/dk)* (k) . dY*"
    """
    _solver.f90wrap_gradient_test(setup=self._handle, mesh=mesh._handle, \
        input_data=input_data._handle, parameters=parameters._handle, \
        states=states._handle, output=output._handle)


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module \
        "mw_adjoint_test".')

for func in _dt_array_initialisers:
    func()
