"""
Module mwd_mesh


Defined at smash/solver/module/mwd_mesh.f90 lines 38-101

This module `mwd_mesh` encapsulates all SMASH mesh.
This module is wrapped and differentiated.

MeshDT type:

</> Public
======================== =======================================
`Variables`              Description
======================== =======================================
``dx``                   Solver spatial step                 [m]
``nrow``                 Number of row
``ncol``                 Number of column
``ng``                   Number of gauge
``nac``                  Number of active cell
``xmin``                 CRS x mininimum value               [m]
``ymax``                 CRS y maximum value                 [m]
``flwdir``               Flow directions
``drained_area``         Drained area                        [nb of cell]
``path``                 Solver path
``active_cell``          Mask of active cell
``flwdst``               Flow distances from main outlet(s)  [m]
``gauge_pos``            Gauge position
``code``                 Gauge code
``area``                 Drained area at gauge position      [m2]

</> Private
======================== =======================================
`Variables`              Description
======================== =======================================
``wgauge``               Objective function gauge weight
``rowcol_to_ind_sparse`` Matrix linking(row, col) couple to sparse storage \
    indice(k)
``local_active_cell``    Mask of local active cell(\in active_cell)
======================== =======================================

contains

[1] MeshDT_initialise
"""
from __future__ import print_function, absolute_import, division
from smash.solver._mw_routine import copy_mesh
from smash.solver import _solver
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("solver.MeshDT")
class MeshDT(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=meshdt)
    
    
    Defined at smash/solver/module/mwd_mesh.f90 lines 42-62
    
    </> Public
    """
    def __init__(self, setup, nrow, ncol, ng, handle=None):
        """
        self = Meshdt(setup, nrow, ncol, ng)
        
        
        Defined at smash/solver/module/mwd_mesh.f90 lines 65-101
        
        Parameters
        ----------
        setup : Setupdt
        nrow : int
        ncol : int
        ng : int
        
        Returns
        -------
        mesh : Meshdt
        
        Notes
        -----
        
        MeshDT initialisation subroutine
        only: sp
        only: SetupDT
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        result = _solver.f90wrap_meshdt_initialise(setup=setup._handle, nrow=nrow, \
            ncol=ncol, ng=ng)
        self._handle = result[0] if isinstance(result, tuple) else result
    
    def __del__(self):
        """
        Destructor for class Meshdt
        
        
        Defined at smash/solver/module/mwd_mesh.f90 lines 42-62
        
        Parameters
        ----------
        this : Meshdt
            Object to be destructed
        
        
        Automatically generated destructor for meshdt
        """
        if self._alloc:
            try:
                _solver.f90wrap_meshdt_finalise(this=self._handle)
            except:
                pass
    
    @property
    def dx(self):
        """
        Element dx ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 44
        
        """
        return _solver.f90wrap_meshdt__get__dx(self._handle)
    
    @dx.setter
    def dx(self, dx):
        _solver.f90wrap_meshdt__set__dx(self._handle, dx)
    
    @property
    def nrow(self):
        """
        Element nrow ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 45
        
        """
        return _solver.f90wrap_meshdt__get__nrow(self._handle)
    
    @nrow.setter
    def nrow(self, nrow):
        _solver.f90wrap_meshdt__set__nrow(self._handle, nrow)
    
    @property
    def ncol(self):
        """
        Element ncol ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 46
        
        """
        return _solver.f90wrap_meshdt__get__ncol(self._handle)
    
    @ncol.setter
    def ncol(self, ncol):
        _solver.f90wrap_meshdt__set__ncol(self._handle, ncol)
    
    @property
    def ng(self):
        """
        Element ng ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 47
        
        """
        return _solver.f90wrap_meshdt__get__ng(self._handle)
    
    @ng.setter
    def ng(self, ng):
        _solver.f90wrap_meshdt__set__ng(self._handle, ng)
    
    @property
    def nac(self):
        """
        Element nac ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 48
        
        """
        return _solver.f90wrap_meshdt__get__nac(self._handle)
    
    @nac.setter
    def nac(self, nac):
        _solver.f90wrap_meshdt__set__nac(self._handle, nac)
    
    @property
    def xmin(self):
        """
        Element xmin ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 49
        
        """
        return _solver.f90wrap_meshdt__get__xmin(self._handle)
    
    @xmin.setter
    def xmin(self, xmin):
        _solver.f90wrap_meshdt__set__xmin(self._handle, xmin)
    
    @property
    def ymax(self):
        """
        Element ymax ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 50
        
        """
        return _solver.f90wrap_meshdt__get__ymax(self._handle)
    
    @ymax.setter
    def ymax(self, ymax):
        _solver.f90wrap_meshdt__set__ymax(self._handle, ymax)
    
    @property
    def flwdir(self):
        """
        Element flwdir ftype=integer pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 51
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_meshdt__array__flwdir(self._handle)
        if array_handle in self._arrays:
            flwdir = self._arrays[array_handle]
        else:
            flwdir = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_meshdt__array__flwdir)
            self._arrays[array_handle] = flwdir
        return flwdir
    
    @flwdir.setter
    def flwdir(self, flwdir):
        self.flwdir[...] = flwdir
    
    @property
    def drained_area(self):
        """
        Element drained_area ftype=integer pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 52
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_meshdt__array__drained_area(self._handle)
        if array_handle in self._arrays:
            drained_area = self._arrays[array_handle]
        else:
            drained_area = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_meshdt__array__drained_area)
            self._arrays[array_handle] = drained_area
        return drained_area
    
    @drained_area.setter
    def drained_area(self, drained_area):
        self.drained_area[...] = drained_area
    
    @property
    def path(self):
        """
        Element path ftype=integer pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 53
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_meshdt__array__path(self._handle)
        if array_handle in self._arrays:
            path = self._arrays[array_handle]
        else:
            path = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_meshdt__array__path)
            self._arrays[array_handle] = path
        return path
    
    @path.setter
    def path(self, path):
        self.path[...] = path
    
    @property
    def active_cell(self):
        """
        Element active_cell ftype=integer pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 54
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_meshdt__array__active_cell(self._handle)
        if array_handle in self._arrays:
            active_cell = self._arrays[array_handle]
        else:
            active_cell = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_meshdt__array__active_cell)
            self._arrays[array_handle] = active_cell
        return active_cell
    
    @active_cell.setter
    def active_cell(self, active_cell):
        self.active_cell[...] = active_cell
    
    @property
    def flwdst(self):
        """
        Element flwdst ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 55
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_meshdt__array__flwdst(self._handle)
        if array_handle in self._arrays:
            flwdst = self._arrays[array_handle]
        else:
            flwdst = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_meshdt__array__flwdst)
            self._arrays[array_handle] = flwdst
        return flwdst
    
    @flwdst.setter
    def flwdst(self, flwdst):
        self.flwdst[...] = flwdst
    
    @property
    def gauge_pos(self):
        """
        Element gauge_pos ftype=integer pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 56
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_meshdt__array__gauge_pos(self._handle)
        if array_handle in self._arrays:
            gauge_pos = self._arrays[array_handle]
        else:
            gauge_pos = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_meshdt__array__gauge_pos)
            self._arrays[array_handle] = gauge_pos
        return gauge_pos
    
    @gauge_pos.setter
    def gauge_pos(self, gauge_pos):
        self.gauge_pos[...] = gauge_pos
    
    @property
    def code(self):
        """
        Element code ftype=character(20) pytype=str
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 57
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_meshdt__array__code(self._handle)
        if array_handle in self._arrays:
            code = self._arrays[array_handle]
        else:
            code = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_meshdt__array__code)
            self._arrays[array_handle] = code
        return code
    
    @code.setter
    def code(self, code):
        self.code[...] = code
    
    @property
    def area(self):
        """
        Element area ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 58
        
        </> Private
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_meshdt__array__area(self._handle)
        if array_handle in self._arrays:
            area = self._arrays[array_handle]
        else:
            area = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_meshdt__array__area)
            self._arrays[array_handle] = area
        return area
    
    @area.setter
    def area(self, area):
        self.area[...] = area
    
    @property
    def _wgauge(self):
        """
        Element wgauge ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 60
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_meshdt__array__wgauge(self._handle)
        if array_handle in self._arrays:
            wgauge = self._arrays[array_handle]
        else:
            wgauge = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_meshdt__array__wgauge)
            self._arrays[array_handle] = wgauge
        return wgauge
    
    @_wgauge.setter
    def _wgauge(self, wgauge):
        self._wgauge[...] = wgauge
    
    @property
    def _rowcol_to_ind_sparse(self):
        """
        Element rowcol_to_ind_sparse ftype=integer pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 61
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_meshdt__array__rowcol_to_ind_sparse(self._handle)
        if array_handle in self._arrays:
            rowcol_to_ind_sparse = self._arrays[array_handle]
        else:
            rowcol_to_ind_sparse = \
                f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_meshdt__array__rowcol_to_ind_sparse)
            self._arrays[array_handle] = rowcol_to_ind_sparse
        return rowcol_to_ind_sparse
    
    @_rowcol_to_ind_sparse.setter
    def _rowcol_to_ind_sparse(self, rowcol_to_ind_sparse):
        self._rowcol_to_ind_sparse[...] = rowcol_to_ind_sparse
    
    @property
    def _local_active_cell(self):
        """
        Element local_active_cell ftype=integer pytype=int
        
        
        Defined at smash/solver/module/mwd_mesh.f90 line 62
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_meshdt__array__local_active_cell(self._handle)
        if array_handle in self._arrays:
            local_active_cell = self._arrays[array_handle]
        else:
            local_active_cell = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_meshdt__array__local_active_cell)
            self._arrays[array_handle] = local_active_cell
        return local_active_cell
    
    @_local_active_cell.setter
    def _local_active_cell(self, local_active_cell):
        self._local_active_cell[...] = local_active_cell
    
    def __str__(self):
        ret = ['<meshdt>{\n']
        ret.append('    dx : ')
        ret.append(repr(self.dx))
        ret.append(',\n    nrow : ')
        ret.append(repr(self.nrow))
        ret.append(',\n    ncol : ')
        ret.append(repr(self.ncol))
        ret.append(',\n    ng : ')
        ret.append(repr(self.ng))
        ret.append(',\n    nac : ')
        ret.append(repr(self.nac))
        ret.append(',\n    xmin : ')
        ret.append(repr(self.xmin))
        ret.append(',\n    ymax : ')
        ret.append(repr(self.ymax))
        ret.append(',\n    flwdir : ')
        ret.append(repr(self.flwdir))
        ret.append(',\n    drained_area : ')
        ret.append(repr(self.drained_area))
        ret.append(',\n    path : ')
        ret.append(repr(self.path))
        ret.append(',\n    active_cell : ')
        ret.append(repr(self.active_cell))
        ret.append(',\n    flwdst : ')
        ret.append(repr(self.flwdst))
        ret.append(',\n    gauge_pos : ')
        ret.append(repr(self.gauge_pos))
        ret.append(',\n    code : ')
        ret.append(repr(self.code))
        ret.append(',\n    area : ')
        ret.append(repr(self.area))
        ret.append('}')
        return ''.join(ret)

    def copy(self):
        return copy_mesh(self)
    
    _dt_array_initialisers = []
    


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module "mwd_mesh".')

for func in _dt_array_initialisers:
    func()
