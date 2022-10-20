"""
Module mwd_common


Defined at smash/solver/module/mwd_common.f90 lines 18-38

This module `md_common` encapsulates all SMASH common.
This module is wrapped and differentiated.

mwd_common variables

</> Public
======================= ========================================
`Variables`             Description
======================= ========================================
``sp``                  Single precision value
``dp``                  Double precision value
``lchar``               Characeter length value
``np``                  Number of SMASH parameters(ParametersDT)
``ns``                  Number of SMASH states(StatesDT)
``name_parameters``     Name of SMASH parameters
``name_states``         Name of SMASH states
======================= ========================================
"""
from __future__ import print_function, absolute_import, division
from smash.solver import _solver
import f90wrap.runtime
import logging
import numpy

_arrays = {}
_objs = {}

def get_sp():
    """
    Element sp ftype=integer pytype=int
    
    
    Defined at smash/solver/module/mwd_common.f90 line 20
    
    """
    return _solver.f90wrap_mwd_common__get__sp()

sp = get_sp()

def get_dp():
    """
    Element dp ftype=integer pytype=int
    
    
    Defined at smash/solver/module/mwd_common.f90 line 21
    
    """
    return _solver.f90wrap_mwd_common__get__dp()

dp = get_dp()

def get_lchar():
    """
    Element lchar ftype=integer pytype=int
    
    
    Defined at smash/solver/module/mwd_common.f90 line 22
    
    """
    return _solver.f90wrap_mwd_common__get__lchar()

lchar = get_lchar()

def get_np():
    """
    Element np ftype=integer pytype=int
    
    
    Defined at smash/solver/module/mwd_common.f90 line 23
    
    """
    return _solver.f90wrap_mwd_common__get__np()

np = get_np()

def get_ns():
    """
    Element ns ftype=integer pytype=int
    
    
    Defined at smash/solver/module/mwd_common.f90 line 24
    
    """
    return _solver.f90wrap_mwd_common__get__ns()

ns = get_ns()

def get_array_name_parameters():
    """
    Element name_parameters ftype=character(10) pytype=str
    
    
    Defined at smash/solver/module/mwd_common.f90 line 33
    
    """
    global name_parameters
    array_ndim, array_type, array_shape, array_handle = \
        _solver.f90wrap_mwd_common__array__name_parameters(f90wrap.runtime.empty_handle)
    if array_handle in _arrays:
        name_parameters = _arrays[array_handle]
    else:
        name_parameters = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                f90wrap.runtime.empty_handle,
                                _solver.f90wrap_mwd_common__array__name_parameters)
        _arrays[array_handle] = name_parameters
    name_parameters = numpy.array(name_parameters.tobytes(order='F').decode('utf-8').split())
    return name_parameters

def set_array_name_parameters(name_parameters):
    name_parameters[...] = name_parameters

def get_array_name_states():
    """
    Element name_states ftype=character(10) pytype=str
    
    
    Defined at smash/solver/module/mwd_common.f90 line 39
    
    """
    global name_states
    array_ndim, array_type, array_shape, array_handle = \
        _solver.f90wrap_mwd_common__array__name_states(f90wrap.runtime.empty_handle)
    if array_handle in _arrays:
        name_states = _arrays[array_handle]
    else:
        name_states = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                f90wrap.runtime.empty_handle,
                                _solver.f90wrap_mwd_common__array__name_states)
        _arrays[array_handle] = name_states
    name_states = numpy.array(name_states.tobytes(order='F').decode('utf-8').split())
    return name_states

def set_array_name_states(name_states):
    name_states[...] = name_states


_array_initialisers = [get_array_name_parameters, get_array_name_states]
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module "mwd_common".')

for func in _dt_array_initialisers:
    func()
