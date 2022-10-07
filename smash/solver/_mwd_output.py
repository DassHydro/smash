"""
Module mwd_output


Defined at smash/solver/module/mwd_output.f90 lines 25-82

This module `mwd_output` encapsulates all SMASH output.
This module is wrapped and differentiated.

OutputDT type:

</> Public
======================== =======================================
`Variables`              Description
======================== =======================================
``qsim``                 Simulated discharge at gauge            [m3/s]
``qsim_domain``          Simulated discharge whole domain        [m3/s]
``sparse_qsim_domain``   Sparse simulated discharge whole domain [m3/s]
``parameters_gradient``  Parameters gradients
``cost``                 Cost value
``sp1``                  Scalar product <dY*, dY>
``sp2``                  Scalar product <dk*, dk>
``an``                   Alpha gradient test
``ian``                  Ialpha gradient test
``fstates``              Final states(StatesDT)
======================== =======================================

contains

[1] OutputDT_initialise
"""
from __future__ import print_function, absolute_import, division
from smash.solver._mw_routine import copy_output
from smash.solver import _solver
import f90wrap.runtime
import logging
from smash.solver._mwd_states import StatesDT

_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("solver.OutputDT")
class OutputDT(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=outputdt)
    
    
    Defined at smash/solver/module/mwd_output.f90 lines 31-43
    
    """
    def __init__(self, setup, mesh, handle=None):
        """
        self = Outputdt(setup, mesh)
        
        
        Defined at smash/solver/module/mwd_output.f90 lines 46-82
        
        Parameters
        ----------
        setup : Setupdt
        mesh : Meshdt
        
        Returns
        -------
        output : Outputdt
        
        Notes
        -----
        
        OutputDT initialisation subroutine
        only: sp, dp, lchar, np, ns
        only: SetupDT
        only: MeshDT
        only: StatesDT, StatesDT_initialise
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        result = _solver.f90wrap_outputdt_initialise(setup=setup._handle, \
            mesh=mesh._handle)
        self._handle = result[0] if isinstance(result, tuple) else result
    
    def __del__(self):
        """
        Destructor for class Outputdt
        
        
        Defined at smash/solver/module/mwd_output.f90 lines 31-43
        
        Parameters
        ----------
        this : Outputdt
            Object to be destructed
        
        
        Automatically generated destructor for outputdt
        """
        if self._alloc:
            try:
                _solver.f90wrap_outputdt_finalise(this=self._handle)
            except:
                pass
    
    @property
    def qsim(self):
        """
        Element qsim ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_output.f90 line 32
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_outputdt__array__qsim(self._handle)
        if array_handle in self._arrays:
            qsim = self._arrays[array_handle]
        else:
            qsim = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_outputdt__array__qsim)
            self._arrays[array_handle] = qsim
        return qsim
    
    @qsim.setter
    def qsim(self, qsim):
        self.qsim[...] = qsim
    
    @property
    def qsim_domain(self):
        """
        Element qsim_domain ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_output.f90 line 33
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_outputdt__array__qsim_domain(self._handle)
        if array_handle in self._arrays:
            qsim_domain = self._arrays[array_handle]
        else:
            qsim_domain = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_outputdt__array__qsim_domain)
            self._arrays[array_handle] = qsim_domain
        return qsim_domain
    
    @qsim_domain.setter
    def qsim_domain(self, qsim_domain):
        self.qsim_domain[...] = qsim_domain
    
    @property
    def sparse_qsim_domain(self):
        """
        Element sparse_qsim_domain ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_output.f90 line 34
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_outputdt__array__sparse_qsim_domain(self._handle)
        if array_handle in self._arrays:
            sparse_qsim_domain = self._arrays[array_handle]
        else:
            sparse_qsim_domain = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_outputdt__array__sparse_qsim_domain)
            self._arrays[array_handle] = sparse_qsim_domain
        return sparse_qsim_domain
    
    @sparse_qsim_domain.setter
    def sparse_qsim_domain(self, sparse_qsim_domain):
        self.sparse_qsim_domain[...] = sparse_qsim_domain
    
    @property
    def net_prcp_domain(self):
        """
        Element net_prcp_domain ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_output.f90 line 35
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_outputdt__array__net_prcp_domain(self._handle)
        if array_handle in self._arrays:
            net_prcp_domain = self._arrays[array_handle]
        else:
            net_prcp_domain = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_outputdt__array__net_prcp_domain)
            self._arrays[array_handle] = net_prcp_domain
        return net_prcp_domain
    
    @net_prcp_domain.setter
    def net_prcp_domain(self, net_prcp_domain):
        self.net_prcp_domain[...] = net_prcp_domain
    
    @property
    def sparse_net_prcp_domain(self):
        """
        Element sparse_net_prcp_domain ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_output.f90 line 36
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_outputdt__array__sparse_net_prcp_domain(self._handle)
        if array_handle in self._arrays:
            sparse_net_prcp_domain = self._arrays[array_handle]
        else:
            sparse_net_prcp_domain = \
                f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_outputdt__array__sparse_net_prcp_domain)
            self._arrays[array_handle] = sparse_net_prcp_domain
        return sparse_net_prcp_domain
    
    @sparse_net_prcp_domain.setter
    def sparse_net_prcp_domain(self, sparse_net_prcp_domain):
        self.sparse_net_prcp_domain[...] = sparse_net_prcp_domain
    
    @property
    def parameters_gradient(self):
        """
        Element parameters_gradient ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_output.f90 line 37
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_outputdt__array__parameters_gradient(self._handle)
        if array_handle in self._arrays:
            parameters_gradient = self._arrays[array_handle]
        else:
            parameters_gradient = \
                f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_outputdt__array__parameters_gradient)
            self._arrays[array_handle] = parameters_gradient
        return parameters_gradient
    
    @parameters_gradient.setter
    def parameters_gradient(self, parameters_gradient):
        self.parameters_gradient[...] = parameters_gradient
    
    @property
    def cost(self):
        """
        Element cost ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_output.f90 line 38
        
        """
        return _solver.f90wrap_outputdt__get__cost(self._handle)
    
    @cost.setter
    def cost(self, cost):
        _solver.f90wrap_outputdt__set__cost(self._handle, cost)
    
    @property
    def sp1(self):
        """
        Element sp1 ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_output.f90 line 39
        
        """
        return _solver.f90wrap_outputdt__get__sp1(self._handle)
    
    @sp1.setter
    def sp1(self, sp1):
        _solver.f90wrap_outputdt__set__sp1(self._handle, sp1)
    
    @property
    def sp2(self):
        """
        Element sp2 ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_output.f90 line 40
        
        """
        return _solver.f90wrap_outputdt__get__sp2(self._handle)
    
    @sp2.setter
    def sp2(self, sp2):
        _solver.f90wrap_outputdt__set__sp2(self._handle, sp2)
    
    @property
    def an(self):
        """
        Element an ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_output.f90 line 41
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_outputdt__array__an(self._handle)
        if array_handle in self._arrays:
            an = self._arrays[array_handle]
        else:
            an = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_outputdt__array__an)
            self._arrays[array_handle] = an
        return an
    
    @an.setter
    def an(self, an):
        self.an[...] = an
    
    @property
    def ian(self):
        """
        Element ian ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_output.f90 line 42
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_outputdt__array__ian(self._handle)
        if array_handle in self._arrays:
            ian = self._arrays[array_handle]
        else:
            ian = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_outputdt__array__ian)
            self._arrays[array_handle] = ian
        return ian
    
    @ian.setter
    def ian(self, ian):
        self.ian[...] = ian
    
    @property
    def fstates(self):
        """
        Element fstates ftype=type(statesdt) pytype=Statesdt
        
        
        Defined at smash/solver/module/mwd_output.f90 line 43
        
        """
        fstates_handle = _solver.f90wrap_outputdt__get__fstates(self._handle)
        if tuple(fstates_handle) in self._objs:
            fstates = self._objs[tuple(fstates_handle)]
        else:
            fstates = StatesDT.from_handle(fstates_handle)
            self._objs[tuple(fstates_handle)] = fstates
        return fstates
    
    @fstates.setter
    def fstates(self, fstates):
        fstates = fstates._handle
        _solver.f90wrap_outputdt__set__fstates(self._handle, fstates)
    
    def __str__(self):
        ret = ['<outputdt>{\n']
        ret.append('    qsim : ')
        ret.append(repr(self.qsim))
        ret.append(',\n    qsim_domain : ')
        ret.append(repr(self.qsim_domain))
        ret.append(',\n    sparse_qsim_domain : ')
        ret.append(repr(self.sparse_qsim_domain))
        ret.append(',\n    net_prcp_domain : ')
        ret.append(repr(self.net_prcp_domain))
        ret.append(',\n    sparse_net_prcp_domain : ')
        ret.append(repr(self.sparse_net_prcp_domain))
        ret.append(',\n    parameters_gradient : ')
        ret.append(repr(self.parameters_gradient))
        ret.append(',\n    cost : ')
        ret.append(repr(self.cost))
        ret.append(',\n    sp1 : ')
        ret.append(repr(self.sp1))
        ret.append(',\n    sp2 : ')
        ret.append(repr(self.sp2))
        ret.append(',\n    an : ')
        ret.append(repr(self.an))
        ret.append(',\n    ian : ')
        ret.append(repr(self.ian))
        ret.append(',\n    fstates : ')
        ret.append(repr(self.fstates))
        ret.append('}')
        return ''.join(ret)

    def copy(self):
        return copy_output(self)
    
    _dt_array_initialisers = []
    


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module "mwd_output".')

for func in _dt_array_initialisers:
    func()
