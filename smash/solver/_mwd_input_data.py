"""
Module mwd_input_data


Defined at smash/solver/module/mwd_input_data.f90 lines 44-151

This module `mwd_input_data` encapsulates all SMASH input_data.
This module is wrapped and differentiated.

Prcp_IndiceDT type:

</> Public
======================== =======================================
`Variables`              Description
======================== =======================================
``p0``                   The 0-th spatial moment of catchment prcp  [mm/dt]
``p1``                   The 1-th spatial moment of catchment prpc  [mm2/dt]
``p2``                   The 2-th spatial moment of catchment prpc  [mm3/dt]
``g1``                   The 1-th spatial moment of flow distance   [mm]
``g2``                   The 2-th spatial moment of flow distance   [mm2]
``md1``                  The 1-th scaled moment                     [-]
``md2``                  The 2-th scaled moment                     [-]
``std``                  Standard deviation of catchment prcp       [mm/dt]
``wf``                   Width function                             [-]
``pwf``                  Precipitation width function               [-]
``vg``                   Vertical gap                               [-]
======================== =======================================

Input_DataDT type:

</> Public
======================== =======================================
`Variables`              Description
======================== =======================================
``qobs``                 Oberserved discharge at gauge               [m3/s]
``prcp``                 Precipitation field                         [mm]
``pet``                  Potential evapotranspiration field          [mm]
``descriptor`` Descriptor map(s) field [(descriptor dependent)]
``sparse_prcp``          Sparse precipitation field                  [mm]
``sparse_pet``           Spase potential evapotranspiration field    [mm]
``mean_prcp``            Mean precipitation at gauge                 [mm]
``mean_pet``             Mean potential evapotranspiration at gauge  [mm]
``prcp_indice``          Precipitation indices(Prcp_IndicesDT)
======================== =======================================

contains

[1] Prcp_IndiceDT_initialise
[2] Input_DataDT_initialise
"""
from __future__ import print_function, absolute_import, division
from smash.solver._mw_routine import copy_input_data
from smash.solver import _solver
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("solver.Prcp_IndiceDT")
class Prcp_IndiceDT(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=prcp_indicedt)
    
    
    Defined at smash/solver/module/mwd_input_data.f90 lines 49-61
    
    """
    def __init__(self, setup, mesh, handle=None):
        """
        self = Prcp_Indicedt(setup, mesh)
        
        
        Defined at smash/solver/module/mwd_input_data.f90 lines 75-107
        
        Parameters
        ----------
        setup : Setupdt
        mesh : Meshdt
        
        Returns
        -------
        prcp_indice : Prcp_Indicedt
        
        Notes
        -----
        
        Prcp_IndiceDT initialisation subroutine
        only: sp
        only: SetupDT
        only: MeshDT, mask_gauge
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        result = _solver.f90wrap_prcp_indicedt_initialise(setup=setup._handle, \
            mesh=mesh._handle)
        self._handle = result[0] if isinstance(result, tuple) else result
    
    def __del__(self):
        """
        Destructor for class Prcp_Indicedt
        
        
        Defined at smash/solver/module/mwd_input_data.f90 lines 49-61
        
        Parameters
        ----------
        this : Prcp_Indicedt
            Object to be destructed
        
        
        Automatically generated destructor for prcp_indicedt
        """
        if self._alloc:
            try:
                _solver.f90wrap_prcp_indicedt_finalise(this=self._handle)
            except:
                pass
    
    @property
    def p0(self):
        """
        Element p0 ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 50
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__p0(self._handle)
        if array_handle in self._arrays:
            p0 = self._arrays[array_handle]
        else:
            p0 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__p0)
            self._arrays[array_handle] = p0
        return p0
    
    @p0.setter
    def p0(self, p0):
        self.p0[...] = p0
    
    @property
    def p1(self):
        """
        Element p1 ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 51
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__p1(self._handle)
        if array_handle in self._arrays:
            p1 = self._arrays[array_handle]
        else:
            p1 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__p1)
            self._arrays[array_handle] = p1
        return p1
    
    @p1.setter
    def p1(self, p1):
        self.p1[...] = p1
    
    @property
    def p2(self):
        """
        Element p2 ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 52
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__p2(self._handle)
        if array_handle in self._arrays:
            p2 = self._arrays[array_handle]
        else:
            p2 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__p2)
            self._arrays[array_handle] = p2
        return p2
    
    @p2.setter
    def p2(self, p2):
        self.p2[...] = p2
    
    @property
    def g1(self):
        """
        Element g1 ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 53
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__g1(self._handle)
        if array_handle in self._arrays:
            g1 = self._arrays[array_handle]
        else:
            g1 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__g1)
            self._arrays[array_handle] = g1
        return g1
    
    @g1.setter
    def g1(self, g1):
        self.g1[...] = g1
    
    @property
    def g2(self):
        """
        Element g2 ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 54
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__g2(self._handle)
        if array_handle in self._arrays:
            g2 = self._arrays[array_handle]
        else:
            g2 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__g2)
            self._arrays[array_handle] = g2
        return g2
    
    @g2.setter
    def g2(self, g2):
        self.g2[...] = g2
    
    @property
    def md1(self):
        """
        Element md1 ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 55
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__md1(self._handle)
        if array_handle in self._arrays:
            md1 = self._arrays[array_handle]
        else:
            md1 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__md1)
            self._arrays[array_handle] = md1
        return md1
    
    @md1.setter
    def md1(self, md1):
        self.md1[...] = md1
    
    @property
    def md2(self):
        """
        Element md2 ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 56
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__md2(self._handle)
        if array_handle in self._arrays:
            md2 = self._arrays[array_handle]
        else:
            md2 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__md2)
            self._arrays[array_handle] = md2
        return md2
    
    @md2.setter
    def md2(self, md2):
        self.md2[...] = md2
    
    @property
    def std(self):
        """
        Element std ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 57
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__std(self._handle)
        if array_handle in self._arrays:
            std = self._arrays[array_handle]
        else:
            std = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__std)
            self._arrays[array_handle] = std
        return std
    
    @std.setter
    def std(self, std):
        self.std[...] = std
    
    @property
    def flwdst_qtl(self):
        """
        Element flwdst_qtl ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 58
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__flwdst_qtl(self._handle)
        if array_handle in self._arrays:
            flwdst_qtl = self._arrays[array_handle]
        else:
            flwdst_qtl = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__flwdst_qtl)
            self._arrays[array_handle] = flwdst_qtl
        return flwdst_qtl
    
    @flwdst_qtl.setter
    def flwdst_qtl(self, flwdst_qtl):
        self.flwdst_qtl[...] = flwdst_qtl
    
    @property
    def wf(self):
        """
        Element wf ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 59
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__wf(self._handle)
        if array_handle in self._arrays:
            wf = self._arrays[array_handle]
        else:
            wf = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__wf)
            self._arrays[array_handle] = wf
        return wf
    
    @wf.setter
    def wf(self, wf):
        self.wf[...] = wf
    
    @property
    def pwf(self):
        """
        Element pwf ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 60
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__pwf(self._handle)
        if array_handle in self._arrays:
            pwf = self._arrays[array_handle]
        else:
            pwf = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__pwf)
            self._arrays[array_handle] = pwf
        return pwf
    
    @pwf.setter
    def pwf(self, pwf):
        self.pwf[...] = pwf
    
    @property
    def vg(self):
        """
        Element vg ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 61
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_prcp_indicedt__array__vg(self._handle)
        if array_handle in self._arrays:
            vg = self._arrays[array_handle]
        else:
            vg = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_prcp_indicedt__array__vg)
            self._arrays[array_handle] = vg
        return vg
    
    @vg.setter
    def vg(self, vg):
        self.vg[...] = vg
    
    def __str__(self):
        ret = ['<prcp_indicedt>{\n']
        ret.append('    p0 : ')
        ret.append(repr(self.p0))
        ret.append(',\n    p1 : ')
        ret.append(repr(self.p1))
        ret.append(',\n    p2 : ')
        ret.append(repr(self.p2))
        ret.append(',\n    g1 : ')
        ret.append(repr(self.g1))
        ret.append(',\n    g2 : ')
        ret.append(repr(self.g2))
        ret.append(',\n    md1 : ')
        ret.append(repr(self.md1))
        ret.append(',\n    md2 : ')
        ret.append(repr(self.md2))
        ret.append(',\n    std : ')
        ret.append(repr(self.std))
        ret.append(',\n    flwdst_qtl : ')
        ret.append(repr(self.flwdst_qtl))
        ret.append(',\n    wf : ')
        ret.append(repr(self.wf))
        ret.append(',\n    pwf : ')
        ret.append(repr(self.pwf))
        ret.append(',\n    vg : ')
        ret.append(repr(self.vg))
        ret.append('}')
        return ''.join(ret)

    def copy(self):
        return copy_input_data(self)
    
    _dt_array_initialisers = []
    

@f90wrap.runtime.register_class("solver.Input_DataDT")
class Input_DataDT(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=input_datadt)
    
    
    Defined at smash/solver/module/mwd_input_data.f90 lines 63-72
    
    """
    def __init__(self, setup, mesh, handle=None):
        """
        self = Input_Datadt(setup, mesh)
        
        
        Defined at smash/solver/module/mwd_input_data.f90 lines 109-151
        
        Parameters
        ----------
        setup : Setupdt
        mesh : Meshdt
        
        Returns
        -------
        input_data : Input_Datadt
        
        Notes
        -----
        
        Input_DataDT initialisation subroutine
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        result = _solver.f90wrap_input_datadt_initialise(setup=setup._handle, \
            mesh=mesh._handle)
        self._handle = result[0] if isinstance(result, tuple) else result
    
    def __del__(self):
        """
        Destructor for class Input_Datadt
        
        
        Defined at smash/solver/module/mwd_input_data.f90 lines 63-72
        
        Parameters
        ----------
        this : Input_Datadt
            Object to be destructed
        
        
        Automatically generated destructor for input_datadt
        """
        if self._alloc:
            try:
                _solver.f90wrap_input_datadt_finalise(this=self._handle)
            except:
                pass
    
    @property
    def qobs(self):
        """
        Element qobs ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 64
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_input_datadt__array__qobs(self._handle)
        if array_handle in self._arrays:
            qobs = self._arrays[array_handle]
        else:
            qobs = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_input_datadt__array__qobs)
            self._arrays[array_handle] = qobs
        return qobs
    
    @qobs.setter
    def qobs(self, qobs):
        self.qobs[...] = qobs
    
    @property
    def prcp(self):
        """
        Element prcp ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 65
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_input_datadt__array__prcp(self._handle)
        if array_handle in self._arrays:
            prcp = self._arrays[array_handle]
        else:
            prcp = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_input_datadt__array__prcp)
            self._arrays[array_handle] = prcp
        return prcp
    
    @prcp.setter
    def prcp(self, prcp):
        self.prcp[...] = prcp
    
    @property
    def pet(self):
        """
        Element pet ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 66
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_input_datadt__array__pet(self._handle)
        if array_handle in self._arrays:
            pet = self._arrays[array_handle]
        else:
            pet = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_input_datadt__array__pet)
            self._arrays[array_handle] = pet
        return pet
    
    @pet.setter
    def pet(self, pet):
        self.pet[...] = pet
    
    @property
    def descriptor(self):
        """
        Element descriptor ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 67
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_input_datadt__array__descriptor(self._handle)
        if array_handle in self._arrays:
            descriptor = self._arrays[array_handle]
        else:
            descriptor = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_input_datadt__array__descriptor)
            self._arrays[array_handle] = descriptor
        return descriptor
    
    @descriptor.setter
    def descriptor(self, descriptor):
        self.descriptor[...] = descriptor
    
    @property
    def sparse_prcp(self):
        """
        Element sparse_prcp ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 68
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_input_datadt__array__sparse_prcp(self._handle)
        if array_handle in self._arrays:
            sparse_prcp = self._arrays[array_handle]
        else:
            sparse_prcp = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_input_datadt__array__sparse_prcp)
            self._arrays[array_handle] = sparse_prcp
        return sparse_prcp
    
    @sparse_prcp.setter
    def sparse_prcp(self, sparse_prcp):
        self.sparse_prcp[...] = sparse_prcp
    
    @property
    def sparse_pet(self):
        """
        Element sparse_pet ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 69
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_input_datadt__array__sparse_pet(self._handle)
        if array_handle in self._arrays:
            sparse_pet = self._arrays[array_handle]
        else:
            sparse_pet = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_input_datadt__array__sparse_pet)
            self._arrays[array_handle] = sparse_pet
        return sparse_pet
    
    @sparse_pet.setter
    def sparse_pet(self, sparse_pet):
        self.sparse_pet[...] = sparse_pet
    
    @property
    def mean_prcp(self):
        """
        Element mean_prcp ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 70
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_input_datadt__array__mean_prcp(self._handle)
        if array_handle in self._arrays:
            mean_prcp = self._arrays[array_handle]
        else:
            mean_prcp = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_input_datadt__array__mean_prcp)
            self._arrays[array_handle] = mean_prcp
        return mean_prcp
    
    @mean_prcp.setter
    def mean_prcp(self, mean_prcp):
        self.mean_prcp[...] = mean_prcp
    
    @property
    def mean_pet(self):
        """
        Element mean_pet ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 71
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_input_datadt__array__mean_pet(self._handle)
        if array_handle in self._arrays:
            mean_pet = self._arrays[array_handle]
        else:
            mean_pet = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_input_datadt__array__mean_pet)
            self._arrays[array_handle] = mean_pet
        return mean_pet
    
    @mean_pet.setter
    def mean_pet(self, mean_pet):
        self.mean_pet[...] = mean_pet
    
    @property
    def prcp_indice(self):
        """
        Element prcp_indice ftype=type(prcp_indicedt) pytype=Prcp_Indicedt
        
        
        Defined at smash/solver/module/mwd_input_data.f90 line 72
        
        """
        prcp_indice_handle = \
            _solver.f90wrap_input_datadt__get__prcp_indice(self._handle)
        if tuple(prcp_indice_handle) in self._objs:
            prcp_indice = self._objs[tuple(prcp_indice_handle)]
        else:
            prcp_indice = Prcp_IndiceDT.from_handle(prcp_indice_handle)
            self._objs[tuple(prcp_indice_handle)] = prcp_indice
        return prcp_indice
    
    @prcp_indice.setter
    def prcp_indice(self, prcp_indice):
        prcp_indice = prcp_indice._handle
        _solver.f90wrap_input_datadt__set__prcp_indice(self._handle, prcp_indice)
    
    def __str__(self):
        ret = ['<input_datadt>{\n']
        ret.append('    qobs : ')
        ret.append(repr(self.qobs))
        ret.append(',\n    prcp : ')
        ret.append(repr(self.prcp))
        ret.append(',\n    pet : ')
        ret.append(repr(self.pet))
        ret.append(',\n    descriptor : ')
        ret.append(repr(self.descriptor))
        ret.append(',\n    sparse_prcp : ')
        ret.append(repr(self.sparse_prcp))
        ret.append(',\n    sparse_pet : ')
        ret.append(repr(self.sparse_pet))
        ret.append(',\n    mean_prcp : ')
        ret.append(repr(self.mean_prcp))
        ret.append(',\n    mean_pet : ')
        ret.append(repr(self.mean_pet))
        ret.append(',\n    prcp_indice : ')
        ret.append(repr(self.prcp_indice))
        ret.append('}')
        return ''.join(ret)

    def copy(self):
        return copy_input_data(self)
    
    _dt_array_initialisers = []
    


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module \
        "mwd_input_data".')

for func in _dt_array_initialisers:
    func()
