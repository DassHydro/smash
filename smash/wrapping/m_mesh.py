"""
Module m_mesh


Defined at smash/f90/wrapped_module/m_mesh.f90 lines 2-38

This module `m_mesh` encapsulates all SMASH mesh(type, subroutines, functions)
"""
from __future__ import print_function, absolute_import, division
import _wrapping
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("wrapping.MeshDT")
class MeshDT(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=meshdt)
    
    
    Defined at smash/f90/wrapped_module/m_mesh.f90 lines 13-22
    
    """
    def __init__(self, setup, nbx, nby, nbc, handle=None):
        """
        self = Meshdt(setup, nbx, nby, nbc)
        
        
        Defined at smash/f90/wrapped_module/m_mesh.f90 lines 25-38
        
        Parameters
        ----------
        setup : Setupdt
        nbx : int
        nby : int
        nbc : int
        
        Returns
        -------
        mesh : Meshdt
        
        MeshDT type:
        
        ==================== ==========================================================
        `args`                  Description
        ==================== ==========================================================
        ==================== ==========================================================
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        result = _wrapping.f90wrap_meshdt_initialise(setup=setup._handle, nbx=nbx, \
            nby=nby, nbc=nbc)
        self._handle = result[0] if isinstance(result, tuple) else result
    
    def __del__(self):
        """
        Destructor for class Meshdt
        
        
        Defined at smash/f90/wrapped_module/m_mesh.f90 lines 13-22
        
        Parameters
        ----------
        this : Meshdt
        	Object to be destructed
        
        
        Automatically generated destructor for meshdt
        """
        if self._alloc:
            _wrapping.f90wrap_meshdt_finalise(this=self._handle)
    
    @property
    def nbx(self):
        """
        Element nbx ftype=integer  pytype=int
        
        
        Defined at smash/f90/wrapped_module/m_mesh.f90 line 14
        
        """
        return _wrapping.f90wrap_meshdt__get__nbx(self._handle)
    
    @nbx.setter
    def nbx(self, nbx):
        _wrapping.f90wrap_meshdt__set__nbx(self._handle, nbx)
    
    @property
    def nby(self):
        """
        Element nby ftype=integer  pytype=int
        
        
        Defined at smash/f90/wrapped_module/m_mesh.f90 line 15
        
        """
        return _wrapping.f90wrap_meshdt__get__nby(self._handle)
    
    @nby.setter
    def nby(self, nby):
        _wrapping.f90wrap_meshdt__set__nby(self._handle, nby)
    
    @property
    def nbc(self):
        """
        Element nbc ftype=integer  pytype=int
        
        
        Defined at smash/f90/wrapped_module/m_mesh.f90 line 16
        
        """
        return _wrapping.f90wrap_meshdt__get__nbc(self._handle)
    
    @nbc.setter
    def nbc(self, nbc):
        _wrapping.f90wrap_meshdt__set__nbc(self._handle, nbc)
    
    @property
    def xll(self):
        """
        Element xll ftype=integer  pytype=int
        
        
        Defined at smash/f90/wrapped_module/m_mesh.f90 line 17
        
        """
        return _wrapping.f90wrap_meshdt__get__xll(self._handle)
    
    @xll.setter
    def xll(self, xll):
        _wrapping.f90wrap_meshdt__set__xll(self._handle, xll)
    
    @property
    def yll(self):
        """
        Element yll ftype=integer  pytype=int
        
        
        Defined at smash/f90/wrapped_module/m_mesh.f90 line 18
        
        """
        return _wrapping.f90wrap_meshdt__get__yll(self._handle)
    
    @yll.setter
    def yll(self, yll):
        _wrapping.f90wrap_meshdt__set__yll(self._handle, yll)
    
    @property
    def flow(self):
        """
        Element flow ftype=integer pytype=int
        
        
        Defined at smash/f90/wrapped_module/m_mesh.f90 line 19
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _wrapping.f90wrap_meshdt__array__flow(self._handle)
        if array_handle in self._arrays:
            flow = self._arrays[array_handle]
        else:
            flow = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _wrapping.f90wrap_meshdt__array__flow)
            self._arrays[array_handle] = flow
        return flow
    
    @flow.setter
    def flow(self, flow):
        self.flow[...] = flow
    
    @property
    def drained_area(self):
        """
        Element drained_area ftype=integer pytype=int
        
        
        Defined at smash/f90/wrapped_module/m_mesh.f90 line 20
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _wrapping.f90wrap_meshdt__array__drained_area(self._handle)
        if array_handle in self._arrays:
            drained_area = self._arrays[array_handle]
        else:
            drained_area = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _wrapping.f90wrap_meshdt__array__drained_area)
            self._arrays[array_handle] = drained_area
        return drained_area
    
    @drained_area.setter
    def drained_area(self, drained_area):
        self.drained_area[...] = drained_area
    
    @property
    def code(self):
        """
        Element code ftype=character(20) pytype=str
        
        
        Defined at smash/f90/wrapped_module/m_mesh.f90 line 21
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _wrapping.f90wrap_meshdt__array__code(self._handle)
        if array_handle in self._arrays:
            code = self._arrays[array_handle]
        else:
            code = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _wrapping.f90wrap_meshdt__array__code)
            self._arrays[array_handle] = code
        return code
    
    @code.setter
    def code(self, code):
        self.code[...] = code
    
    @property
    def area(self):
        """
        Element area ftype=real(dp) pytype=float
        
        
        Defined at smash/f90/wrapped_module/m_mesh.f90 line 22
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _wrapping.f90wrap_meshdt__array__area(self._handle)
        if array_handle in self._arrays:
            area = self._arrays[array_handle]
        else:
            area = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _wrapping.f90wrap_meshdt__array__area)
            self._arrays[array_handle] = area
        return area
    
    @area.setter
    def area(self, area):
        self.area[...] = area
    
    def __str__(self):
        ret = ['<meshdt>{\n']
        ret.append('    nbx : ')
        ret.append(repr(self.nbx))
        ret.append(',\n    nby : ')
        ret.append(repr(self.nby))
        ret.append(',\n    nbc : ')
        ret.append(repr(self.nbc))
        ret.append(',\n    xll : ')
        ret.append(repr(self.xll))
        ret.append(',\n    yll : ')
        ret.append(repr(self.yll))
        ret.append(',\n    flow : ')
        ret.append(repr(self.flow))
        ret.append(',\n    drained_area : ')
        ret.append(repr(self.drained_area))
        ret.append(',\n    code : ')
        ret.append(repr(self.code))
        ret.append(',\n    area : ')
        ret.append(repr(self.area))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module "m_mesh".')

for func in _dt_array_initialisers:
    func()
