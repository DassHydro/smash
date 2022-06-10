"""
Module m_setup


Defined at smash/solver/wrapped_module/m_setup.f90 lines 2-26

This module `m_setup` encapsulates all SMASH setup(type, subroutines, functions)
"""
from __future__ import print_function, absolute_import, division
import _wrapping
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("wrapping.SetupDT")
class SetupDT(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=setupdt)
    
    
    Defined at smash/solver/wrapped_module/m_setup.f90 lines 19-26
    
    """
    def __init__(self, handle=None):
        """
        self = Setupdt()
        
        
        Defined at smash/solver/wrapped_module/m_setup.f90 lines 19-26
        
        
        Returns
        -------
        this : Setupdt
        	Object to be constructed
        
        
        Automatically generated constructor for setupdt
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        result = _wrapping.f90wrap_setupdt_initialise()
        self._handle = result[0] if isinstance(result, tuple) else result
    
    def __del__(self):
        """
        Destructor for class Setupdt
        
        
        Defined at smash/solver/wrapped_module/m_setup.f90 lines 19-26
        
        Parameters
        ----------
        this : Setupdt
        	Object to be destructed
        
        
        Automatically generated destructor for setupdt
        """
        if self._alloc:
            _wrapping.f90wrap_setupdt_finalise(this=self._handle)
    
    @property
    def dt(self):
        """
        Element dt ftype=real(dp) pytype=float
        
        
        Defined at smash/solver/wrapped_module/m_setup.f90 line 20
        
        """
        return _wrapping.f90wrap_setupdt__get__dt(self._handle)
    
    @dt.setter
    def dt(self, dt):
        _wrapping.f90wrap_setupdt__set__dt(self._handle, dt)
    
    @property
    def dx(self):
        """
        Element dx ftype=real(dp) pytype=float
        
        
        Defined at smash/solver/wrapped_module/m_setup.f90 line 21
        
        """
        return _wrapping.f90wrap_setupdt__get__dx(self._handle)
    
    @dx.setter
    def dx(self, dx):
        _wrapping.f90wrap_setupdt__set__dx(self._handle, dx)
    
    @property
    def start_time(self):
        """
        Element start_time ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/wrapped_module/m_setup.f90 line 22
        
        """
        return _wrapping.f90wrap_setupdt__get__start_time(self._handle)
    
    @start_time.setter
    def start_time(self, start_time):
        _wrapping.f90wrap_setupdt__set__start_time(self._handle, start_time)
    
    @property
    def end_time(self):
        """
        Element end_time ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/wrapped_module/m_setup.f90 line 23
        
        """
        return _wrapping.f90wrap_setupdt__get__end_time(self._handle)
    
    @end_time.setter
    def end_time(self, end_time):
        _wrapping.f90wrap_setupdt__set__end_time(self._handle, end_time)
    
    @property
    def optim_start_time(self):
        """
        Element optim_start_time ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/wrapped_module/m_setup.f90 line 24
        
        """
        return _wrapping.f90wrap_setupdt__get__optim_start_time(self._handle)
    
    @optim_start_time.setter
    def optim_start_time(self, optim_start_time):
        _wrapping.f90wrap_setupdt__set__optim_start_time(self._handle, optim_start_time)
    
    @property
    def nb_time_step(self):
        """
        Element nb_time_step ftype=integer  pytype=int
        
        
        Defined at smash/solver/wrapped_module/m_setup.f90 line 25
        
        """
        return _wrapping.f90wrap_setupdt__get__nb_time_step(self._handle)
    
    @nb_time_step.setter
    def nb_time_step(self, nb_time_step):
        _wrapping.f90wrap_setupdt__set__nb_time_step(self._handle, nb_time_step)
    
    @property
    def optim_start_step(self):
        """
        Element optim_start_step ftype=integer  pytype=int
        
        
        Defined at smash/solver/wrapped_module/m_setup.f90 line 26
        
        """
        return _wrapping.f90wrap_setupdt__get__optim_start_step(self._handle)
    
    @optim_start_step.setter
    def optim_start_step(self, optim_start_step):
        _wrapping.f90wrap_setupdt__set__optim_start_step(self._handle, optim_start_step)
    
    def __str__(self):
        ret = ['<setupdt>{\n']
        ret.append('    dt : ')
        ret.append(repr(self.dt))
        ret.append(',\n    dx : ')
        ret.append(repr(self.dx))
        ret.append(',\n    start_time : ')
        ret.append(repr(self.start_time))
        ret.append(',\n    end_time : ')
        ret.append(repr(self.end_time))
        ret.append(',\n    optim_start_time : ')
        ret.append(repr(self.optim_start_time))
        ret.append(',\n    nb_time_step : ')
        ret.append(repr(self.nb_time_step))
        ret.append(',\n    optim_start_step : ')
        ret.append(repr(self.optim_start_step))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module "m_setup".')

for func in _dt_array_initialisers:
    func()
