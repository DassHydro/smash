"""
Module mwd_parameters


Defined at smash/solver/module/mwd_parameters.f90 lines 28-129

This module `mwd_parameters` encapsulates all SMASH parameters.
This module is wrapped and differentiated.

ParametersDT type:

</> Public
======================== =======================================
`Variables`              Description
======================== =======================================
``ci`` Interception parameter [mm] (default: 1) ]0, +Inf[
``cp`` Production parameter [mm] (default: 200) ]0, +Inf[
``beta`` Percolation parameter [-] (default: 1000) ]0, +Inf[
``cft`` Fast transfer parameter [mm] (default: 500) ]0, +Inf[
``cst`` Slow transfer parameter [mm] (default: 500) ]0, +Inf[
``alpha`` Transfer partitioning parameter [-] (default: 0.9) ]0, 1[
``exc`` Exchange parameter [mm/dt] (default: 0) ]-Inf, +Inf[
``lr`` Linear routing parameter [min] (default: 5) ]0, +Inf[
======================== =======================================

contains

[1] ParametersDT_initialise
[2] parameters_to_matrix
[3] matrix_to_parameters
[4] vector_to_parameters
[5] set0_parameters
[6] set1_parameters
"""
from __future__ import print_function, absolute_import, division
from smash.solver._mw_routine import copy_parameters
from smash.solver import _solver
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("solver.ParametersDT")
class ParametersDT(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=parametersdt)
    
    
    Defined at smash/solver/module/mwd_parameters.f90 lines 32-40
    
    """
    def __init__(self, mesh, handle=None):
        """
        self = Parametersdt(mesh)
        
        
        Defined at smash/solver/module/mwd_parameters.f90 lines 43-69
        
        Parameters
        ----------
        mesh : Meshdt
        
        Returns
        -------
        parameters : Parametersdt
        
        Notes
        -----
        
        ParametersDT initialisation subroutine
        only: sp, np
        only: MeshDT
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        result = _solver.f90wrap_parametersdt_initialise(mesh=mesh._handle)
        self._handle = result[0] if isinstance(result, tuple) else result
    
    def __del__(self):
        """
        Destructor for class Parametersdt
        
        
        Defined at smash/solver/module/mwd_parameters.f90 lines 32-40
        
        Parameters
        ----------
        this : Parametersdt
            Object to be destructed
        
        
        Automatically generated destructor for parametersdt
        """
        if self._alloc:
            try:
                _solver.f90wrap_parametersdt_finalise(this=self._handle)
            except:
                pass
    
    @property
    def ci(self):
        """
        Element ci ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_parameters.f90 line 33
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_parametersdt__array__ci(self._handle)
        if array_handle in self._arrays:
            ci = self._arrays[array_handle]
        else:
            ci = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_parametersdt__array__ci)
            self._arrays[array_handle] = ci
        return ci
    
    @ci.setter
    def ci(self, ci):
        self.ci[...] = ci
    
    @property
    def cp(self):
        """
        Element cp ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_parameters.f90 line 34
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_parametersdt__array__cp(self._handle)
        if array_handle in self._arrays:
            cp = self._arrays[array_handle]
        else:
            cp = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_parametersdt__array__cp)
            self._arrays[array_handle] = cp
        return cp
    
    @cp.setter
    def cp(self, cp):
        self.cp[...] = cp
    
    @property
    def beta(self):
        """
        Element beta ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_parameters.f90 line 35
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_parametersdt__array__beta(self._handle)
        if array_handle in self._arrays:
            beta = self._arrays[array_handle]
        else:
            beta = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_parametersdt__array__beta)
            self._arrays[array_handle] = beta
        return beta
    
    @beta.setter
    def beta(self, beta):
        self.beta[...] = beta
    
    @property
    def cft(self):
        """
        Element cft ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_parameters.f90 line 36
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_parametersdt__array__cft(self._handle)
        if array_handle in self._arrays:
            cft = self._arrays[array_handle]
        else:
            cft = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_parametersdt__array__cft)
            self._arrays[array_handle] = cft
        return cft
    
    @cft.setter
    def cft(self, cft):
        self.cft[...] = cft
    
    @property
    def cst(self):
        """
        Element cst ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_parameters.f90 line 37
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_parametersdt__array__cst(self._handle)
        if array_handle in self._arrays:
            cst = self._arrays[array_handle]
        else:
            cst = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_parametersdt__array__cst)
            self._arrays[array_handle] = cst
        return cst
    
    @cst.setter
    def cst(self, cst):
        self.cst[...] = cst
    
    @property
    def alpha(self):
        """
        Element alpha ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_parameters.f90 line 38
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_parametersdt__array__alpha(self._handle)
        if array_handle in self._arrays:
            alpha = self._arrays[array_handle]
        else:
            alpha = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_parametersdt__array__alpha)
            self._arrays[array_handle] = alpha
        return alpha
    
    @alpha.setter
    def alpha(self, alpha):
        self.alpha[...] = alpha
    
    @property
    def exc(self):
        """
        Element exc ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_parameters.f90 line 39
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_parametersdt__array__exc(self._handle)
        if array_handle in self._arrays:
            exc = self._arrays[array_handle]
        else:
            exc = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_parametersdt__array__exc)
            self._arrays[array_handle] = exc
        return exc
    
    @exc.setter
    def exc(self, exc):
        self.exc[...] = exc
    
    @property
    def lr(self):
        """
        Element lr ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_parameters.f90 line 40
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_parametersdt__array__lr(self._handle)
        if array_handle in self._arrays:
            lr = self._arrays[array_handle]
        else:
            lr = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_parametersdt__array__lr)
            self._arrays[array_handle] = lr
        return lr
    
    @lr.setter
    def lr(self, lr):
        self.lr[...] = lr
    
    def __str__(self):
        ret = ['<parametersdt>{\n']
        ret.append('    ci : ')
        ret.append(repr(self.ci))
        ret.append(',\n    cp : ')
        ret.append(repr(self.cp))
        ret.append(',\n    beta : ')
        ret.append(repr(self.beta))
        ret.append(',\n    cft : ')
        ret.append(repr(self.cft))
        ret.append(',\n    cst : ')
        ret.append(repr(self.cst))
        ret.append(',\n    alpha : ')
        ret.append(repr(self.alpha))
        ret.append(',\n    exc : ')
        ret.append(repr(self.exc))
        ret.append(',\n    lr : ')
        ret.append(repr(self.lr))
        ret.append('}')
        return ''.join(ret)

    def copy(self):
        return copy_parameters(self)
    
    _dt_array_initialisers = []
    

def parameters_to_matrix(self, matrix):
    """
    parameters_to_matrix(self, matrix)
    
    
    Defined at smash/solver/module/mwd_parameters.f90 lines 72-84
    
    Parameters
    ----------
    parameters : Parametersdt
    matrix : float array
    
    TODO comment
    """
    _solver.f90wrap_parameters_to_matrix(parameters=self._handle, matrix=matrix)

def matrix_to_parameters(matrix, parameters):
    """
    matrix_to_parameters(matrix, parameters)
    
    
    Defined at smash/solver/module/mwd_parameters.f90 lines 87-99
    
    Parameters
    ----------
    matrix : float array
    parameters : Parametersdt
    
    TODO comment
    """
    _solver.f90wrap_matrix_to_parameters(matrix=matrix, \
        parameters=parameters._handle)

def vector_to_parameters(vector, parameters):
    """
    vector_to_parameters(vector, parameters)
    
    
    Defined at smash/solver/module/mwd_parameters.f90 lines 102-113
    
    Parameters
    ----------
    vector : float array
    parameters : Parametersdt
    
    TODO comment
    """
    _solver.f90wrap_vector_to_parameters(vector=vector, \
        parameters=parameters._handle)

def set0_parameters(self):
    """
    set0_parameters(self)
    
    
    Defined at smash/solver/module/mwd_parameters.f90 lines 116-121
    
    Parameters
    ----------
    parameters : Parametersdt
    
    TODO comment
    """
    _solver.f90wrap_set0_parameters(parameters=self._handle)

def set1_parameters(self):
    """
    set1_parameters(self)
    
    
    Defined at smash/solver/module/mwd_parameters.f90 lines 124-129
    
    Parameters
    ----------
    parameters : Parametersdt
    
    TODO comment
    """
    _solver.f90wrap_set1_parameters(parameters=self._handle)


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module \
        "mwd_parameters".')

for func in _dt_array_initialisers:
    func()
