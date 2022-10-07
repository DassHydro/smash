"""
Module mwd_states


Defined at smash/solver/module/mwd_states.f90 lines 25-104

This module `mwd_states` encapsulates all SMASH states.
This module is wrapped and differentiated.

StatesDT type:

</> Public
======================== =======================================
`Variables`              Description
======================== =======================================
``hi``                   Interception state    [-]   (default: 0.01)   ]0, 1[
``hp``                   Production state      [-]   (default: 0.01)   ]0, 1[
``hft``                  Fast transfer state   [-]   (default: 0.01)   ]0, 1[
``hst``                  Slow transfer state   [-]   (default: 0.01)   ]0, 1[
``hlr`` Linear routing state [mm] (default: 0.01) ]0, +Inf[
======================== =======================================

contains

[1] StatesDT_initialise
[2] states_to_matrix
[3] matrix_to_states
[4] vector_to_states
[5] set0_states
[6] set1_states
"""
from __future__ import print_function, absolute_import, division
from smash.solver._mw_routine import copy_states
from smash.solver import _solver
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("solver.StatesDT")
class StatesDT(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=statesdt)
    
    
    Defined at smash/solver/module/mwd_states.f90 lines 29-34
    
    """
    def __init__(self, mesh, handle=None):
        """
        self = Statesdt(mesh)
        
        
        Defined at smash/solver/module/mwd_states.f90 lines 37-53
        
        Parameters
        ----------
        mesh : Meshdt
        
        Returns
        -------
        states : Statesdt
        
        only: sp, ns
        only: MeshDT
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        result = _solver.f90wrap_statesdt_initialise(mesh=mesh._handle)
        self._handle = result[0] if isinstance(result, tuple) else result
    
    def __del__(self):
        """
        Destructor for class Statesdt
        
        
        Defined at smash/solver/module/mwd_states.f90 lines 29-34
        
        Parameters
        ----------
        this : Statesdt
            Object to be destructed
        
        
        Automatically generated destructor for statesdt
        """
        if self._alloc:
            try:
                _solver.f90wrap_statesdt_finalise(this=self._handle)
            except:
                pass
    
    @property
    def hi(self):
        """
        Element hi ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_states.f90 line 30
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_statesdt__array__hi(self._handle)
        if array_handle in self._arrays:
            hi = self._arrays[array_handle]
        else:
            hi = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_statesdt__array__hi)
            self._arrays[array_handle] = hi
        return hi
    
    @hi.setter
    def hi(self, hi):
        self.hi[...] = hi
    
    @property
    def hp(self):
        """
        Element hp ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_states.f90 line 31
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_statesdt__array__hp(self._handle)
        if array_handle in self._arrays:
            hp = self._arrays[array_handle]
        else:
            hp = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_statesdt__array__hp)
            self._arrays[array_handle] = hp
        return hp
    
    @hp.setter
    def hp(self, hp):
        self.hp[...] = hp
    
    @property
    def hft(self):
        """
        Element hft ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_states.f90 line 32
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_statesdt__array__hft(self._handle)
        if array_handle in self._arrays:
            hft = self._arrays[array_handle]
        else:
            hft = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_statesdt__array__hft)
            self._arrays[array_handle] = hft
        return hft
    
    @hft.setter
    def hft(self, hft):
        self.hft[...] = hft
    
    @property
    def hst(self):
        """
        Element hst ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_states.f90 line 33
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_statesdt__array__hst(self._handle)
        if array_handle in self._arrays:
            hst = self._arrays[array_handle]
        else:
            hst = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_statesdt__array__hst)
            self._arrays[array_handle] = hst
        return hst
    
    @hst.setter
    def hst(self, hst):
        self.hst[...] = hst
    
    @property
    def hlr(self):
        """
        Element hlr ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_states.f90 line 34
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_statesdt__array__hlr(self._handle)
        if array_handle in self._arrays:
            hlr = self._arrays[array_handle]
        else:
            hlr = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_statesdt__array__hlr)
            self._arrays[array_handle] = hlr
        return hlr
    
    @hlr.setter
    def hlr(self, hlr):
        self.hlr[...] = hlr
    
    def __str__(self):
        ret = ['<statesdt>{\n']
        ret.append('    hi : ')
        ret.append(repr(self.hi))
        ret.append(',\n    hp : ')
        ret.append(repr(self.hp))
        ret.append(',\n    hft : ')
        ret.append(repr(self.hft))
        ret.append(',\n    hst : ')
        ret.append(repr(self.hst))
        ret.append(',\n    hlr : ')
        ret.append(repr(self.hlr))
        ret.append('}')
        return ''.join(ret)

    def copy(self):
        return copy_states(self)
    
    _dt_array_initialisers = []
    

def states_to_matrix(self, matrix):
    """
    states_to_matrix(self, matrix)
    
    
    Defined at smash/solver/module/mwd_states.f90 lines 56-65
    
    Parameters
    ----------
    states : Statesdt
    matrix : float array
    
    TODO comment
    """
    _solver.f90wrap_states_to_matrix(states=self._handle, matrix=matrix)

def matrix_to_states(matrix, states):
    """
    matrix_to_states(matrix, states)
    
    
    Defined at smash/solver/module/mwd_states.f90 lines 68-77
    
    Parameters
    ----------
    matrix : float array
    states : Statesdt
    
    TODO comment
    """
    _solver.f90wrap_matrix_to_states(matrix=matrix, states=states._handle)

def vector_to_states(vector, states):
    """
    vector_to_states(vector, states)
    
    
    Defined at smash/solver/module/mwd_states.f90 lines 80-88
    
    Parameters
    ----------
    vector : float array
    states : Statesdt
    
    TODO comment
    """
    _solver.f90wrap_vector_to_states(vector=vector, states=states._handle)

def set0_states(self):
    """
    set0_states(self)
    
    
    Defined at smash/solver/module/mwd_states.f90 lines 91-96
    
    Parameters
    ----------
    states : Statesdt
    
    TODO comment
    """
    _solver.f90wrap_set0_states(states=self._handle)

def set1_states(self):
    """
    set1_states(self)
    
    
    Defined at smash/solver/module/mwd_states.f90 lines 99-104
    
    Parameters
    ----------
    states : Statesdt
    
    TODO comment
    """
    _solver.f90wrap_set1_states(states=self._handle)


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module "mwd_states".')

for func in _dt_array_initialisers:
    func()
