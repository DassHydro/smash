"""
Module mwd_setup


Defined at smash/solver/module/mwd_setup.f90 lines 61-149

This module `mwd_setup` encapsulates all SMASH setup.
This module is wrapped and differentiated.

SetupDT type:

</> Public
========================== =====================================
`Variables`                Description
========================== =====================================
``dt`` Solver time step [s] (default: 3600)
``start_time`` Simulation start time [%Y%m%d%H%M] (default: '...')
``end_time`` Simulation end time [%Y%m%d%H%M] (default: '...')
``sparse_storage``         Forcing sparse storage(default: .false.)
``read_qobs``              Read observed discharge(default: .false.)
``qobs_directory``         Observed discharge directory path(default: '...')
``read_prcp``              Read precipitation(default: .false.)
``prcp_format``            Precipitation format(default: 'tif')
``prcp_conversion_factor`` Precipitation conversion factor(default: 1)
``prcp_directory``         Precipiation directory path(default: '...')
``read_pet``               Reap potential evapotranspiration(default: .false.)
``pet_format``             Potential evapotranspiration format(default: 'tif')
``pet_conversion_factor`` Potential evapotranpisration conversion \
    factor(default: 1)
``pet_directory`` Potential evapotranspiration directory path(default: '...')
``daily_interannual_pet`` Read daily interannual potential \
    evapotranspiration(default: .false.)
``mean_forcing``           Compute mean forcing(default: .false.)
``prcp_indice``            Compute prcp indices(default: .false.)
``read_descriptor`` Read descriptor map(s) (default: .false.)
``descriptor_format``      Descriptor map(s) format(default: .false.)
``descriptor_directory``   Descriptor map(s) directory(default: "...")
``descriptor_name``        Descriptor map(s) names
``interception_module``    Choice of interception module(default: 0)
``production_module``      Choice of production module(default: 0)
``transfer_module``        Choice of transfer module(default: 0)
``exchange_module``        Choice of exchange module(default: 0)
``routing_module``         Choice of routing module(default: 0)
``save_qsim_domain`` Save simulated discharge on the domain(default: .false.)

</> Private
========================== =====================================
`Variables`                Description
========================== =====================================
``ntime_step``             Number of time step
``nd``                     Number of descriptor map(s)
``algorithm``              Optimize Algorithm name
``jobs_fun``               Objective function name(default: 'nse')
``jreg_fun``               Regularization name(default: 'prior')
``wjreg``                  Regularization weight(default: 0)
``optim_start_step``       Optimization start step(default: 1)
``optim_parameters``       Optimized parameters array(default: 0)
``optim_states``           Optimize states array(default: 0)
``lb_parameters``          Parameters lower bounds(default: see below)
``ub_parameters``          Parameters upper bounds(default: see below)
``lb_states``              States lower bounds(default: see below)
``ub_states``              States upper bounds(default: see below)
``maxiter``                Maximum number of iteration(default: 100)
=========================  =====================================

contains

[1] SetupDT_initialise
"""
from __future__ import print_function, absolute_import, division
from smash.solver._mw_routine import copy_setup
from smash.solver import _solver
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("solver.SetupDT")
class SetupDT(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=setupdt)
    
    
    Defined at smash/solver/module/mwd_setup.f90 lines 64-134
    
    </> Public
    """
    def __init__(self, nd, handle=None):
        """
        self = Setupdt(nd)
        
        
        Defined at smash/solver/module/mwd_setup.f90 lines 137-149
        
        Parameters
        ----------
        nd : int
        
        Returns
        -------
        setup : Setupdt
        
        Notes
        -----
        
        SetupDT initialisation subroutine
        only: sp, lchar, np, ns
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        result = _solver.f90wrap_setupdt_initialise(nd=nd)
        self._handle = result[0] if isinstance(result, tuple) else result
    
    def __del__(self):
        """
        Destructor for class Setupdt
        
        
        Defined at smash/solver/module/mwd_setup.f90 lines 64-134
        
        Parameters
        ----------
        this : Setupdt
            Object to be destructed
        
        
        Automatically generated destructor for setupdt
        """
        if self._alloc:
            try:
                _solver.f90wrap_setupdt_finalise(this=self._handle)
            except:
                pass
    
    @property
    def dt(self):
        """
        Element dt ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 66
        
        """
        return _solver.f90wrap_setupdt__get__dt(self._handle)
    
    @dt.setter
    def dt(self, dt):
        _solver.f90wrap_setupdt__set__dt(self._handle, dt)
    
    @property
    def start_time(self):
        """
        Element start_time ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 67
        
        """
        return _solver.f90wrap_setupdt__get__start_time(self._handle)
    
    @start_time.setter
    def start_time(self, start_time):
        _solver.f90wrap_setupdt__set__start_time(self._handle, start_time)
    
    @property
    def end_time(self):
        """
        Element end_time ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 68
        
        """
        return _solver.f90wrap_setupdt__get__end_time(self._handle)
    
    @end_time.setter
    def end_time(self, end_time):
        _solver.f90wrap_setupdt__set__end_time(self._handle, end_time)
    
    @property
    def sparse_storage(self):
        """
        Element sparse_storage ftype=logical pytype=bool
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 69
        
        """
        return _solver.f90wrap_setupdt__get__sparse_storage(self._handle)
    
    @sparse_storage.setter
    def sparse_storage(self, sparse_storage):
        _solver.f90wrap_setupdt__set__sparse_storage(self._handle, sparse_storage)
    
    @property
    def read_qobs(self):
        """
        Element read_qobs ftype=logical pytype=bool
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 70
        
        """
        return _solver.f90wrap_setupdt__get__read_qobs(self._handle)
    
    @read_qobs.setter
    def read_qobs(self, read_qobs):
        _solver.f90wrap_setupdt__set__read_qobs(self._handle, read_qobs)
    
    @property
    def qobs_directory(self):
        """
        Element qobs_directory ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 71
        
        """
        return _solver.f90wrap_setupdt__get__qobs_directory(self._handle)
    
    @qobs_directory.setter
    def qobs_directory(self, qobs_directory):
        _solver.f90wrap_setupdt__set__qobs_directory(self._handle, qobs_directory)
    
    @property
    def read_prcp(self):
        """
        Element read_prcp ftype=logical pytype=bool
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 72
        
        """
        return _solver.f90wrap_setupdt__get__read_prcp(self._handle)
    
    @read_prcp.setter
    def read_prcp(self, read_prcp):
        _solver.f90wrap_setupdt__set__read_prcp(self._handle, read_prcp)
    
    @property
    def prcp_format(self):
        """
        Element prcp_format ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 73
        
        """
        return _solver.f90wrap_setupdt__get__prcp_format(self._handle)
    
    @prcp_format.setter
    def prcp_format(self, prcp_format):
        _solver.f90wrap_setupdt__set__prcp_format(self._handle, prcp_format)
    
    @property
    def prcp_conversion_factor(self):
        """
        Element prcp_conversion_factor ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 74
        
        """
        return _solver.f90wrap_setupdt__get__prcp_conversion_factor(self._handle)
    
    @prcp_conversion_factor.setter
    def prcp_conversion_factor(self, prcp_conversion_factor):
        _solver.f90wrap_setupdt__set__prcp_conversion_factor(self._handle, \
            prcp_conversion_factor)
    
    @property
    def prcp_directory(self):
        """
        Element prcp_directory ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 75
        
        """
        return _solver.f90wrap_setupdt__get__prcp_directory(self._handle)
    
    @prcp_directory.setter
    def prcp_directory(self, prcp_directory):
        _solver.f90wrap_setupdt__set__prcp_directory(self._handle, prcp_directory)
    
    @property
    def read_pet(self):
        """
        Element read_pet ftype=logical pytype=bool
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 76
        
        """
        return _solver.f90wrap_setupdt__get__read_pet(self._handle)
    
    @read_pet.setter
    def read_pet(self, read_pet):
        _solver.f90wrap_setupdt__set__read_pet(self._handle, read_pet)
    
    @property
    def pet_format(self):
        """
        Element pet_format ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 77
        
        """
        return _solver.f90wrap_setupdt__get__pet_format(self._handle)
    
    @pet_format.setter
    def pet_format(self, pet_format):
        _solver.f90wrap_setupdt__set__pet_format(self._handle, pet_format)
    
    @property
    def pet_conversion_factor(self):
        """
        Element pet_conversion_factor ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 78
        
        """
        return _solver.f90wrap_setupdt__get__pet_conversion_factor(self._handle)
    
    @pet_conversion_factor.setter
    def pet_conversion_factor(self, pet_conversion_factor):
        _solver.f90wrap_setupdt__set__pet_conversion_factor(self._handle, \
            pet_conversion_factor)
    
    @property
    def pet_directory(self):
        """
        Element pet_directory ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 79
        
        """
        return _solver.f90wrap_setupdt__get__pet_directory(self._handle)
    
    @pet_directory.setter
    def pet_directory(self, pet_directory):
        _solver.f90wrap_setupdt__set__pet_directory(self._handle, pet_directory)
    
    @property
    def daily_interannual_pet(self):
        """
        Element daily_interannual_pet ftype=logical pytype=bool
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 80
        
        """
        return _solver.f90wrap_setupdt__get__daily_interannual_pet(self._handle)
    
    @daily_interannual_pet.setter
    def daily_interannual_pet(self, daily_interannual_pet):
        _solver.f90wrap_setupdt__set__daily_interannual_pet(self._handle, \
            daily_interannual_pet)
    
    @property
    def mean_forcing(self):
        """
        Element mean_forcing ftype=logical pytype=bool
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 81
        
        """
        return _solver.f90wrap_setupdt__get__mean_forcing(self._handle)
    
    @mean_forcing.setter
    def mean_forcing(self, mean_forcing):
        _solver.f90wrap_setupdt__set__mean_forcing(self._handle, mean_forcing)
    
    @property
    def prcp_indice(self):
        """
        Element prcp_indice ftype=logical pytype=bool
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 82
        
        """
        return _solver.f90wrap_setupdt__get__prcp_indice(self._handle)
    
    @prcp_indice.setter
    def prcp_indice(self, prcp_indice):
        _solver.f90wrap_setupdt__set__prcp_indice(self._handle, prcp_indice)
    
    @property
    def read_descriptor(self):
        """
        Element read_descriptor ftype=logical pytype=bool
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 83
        
        """
        return _solver.f90wrap_setupdt__get__read_descriptor(self._handle)
    
    @read_descriptor.setter
    def read_descriptor(self, read_descriptor):
        _solver.f90wrap_setupdt__set__read_descriptor(self._handle, read_descriptor)
    
    @property
    def descriptor_format(self):
        """
        Element descriptor_format ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 84
        
        """
        return _solver.f90wrap_setupdt__get__descriptor_format(self._handle)
    
    @descriptor_format.setter
    def descriptor_format(self, descriptor_format):
        _solver.f90wrap_setupdt__set__descriptor_format(self._handle, descriptor_format)
    
    @property
    def descriptor_directory(self):
        """
        Element descriptor_directory ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 85
        
        """
        return _solver.f90wrap_setupdt__get__descriptor_directory(self._handle)
    
    @descriptor_directory.setter
    def descriptor_directory(self, descriptor_directory):
        _solver.f90wrap_setupdt__set__descriptor_directory(self._handle, \
            descriptor_directory)
    
    @property
    def descriptor_name(self):
        """
        Element descriptor_name ftype=character(20) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 86
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_setupdt__array__descriptor_name(self._handle)
        if array_handle in self._arrays:
            descriptor_name = self._arrays[array_handle]
        else:
            descriptor_name = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_setupdt__array__descriptor_name)
            self._arrays[array_handle] = descriptor_name
        return descriptor_name
    
    @descriptor_name.setter
    def descriptor_name(self, descriptor_name):
        self.descriptor_name[...] = descriptor_name
    
    @property
    def interception_module(self):
        """
        Element interception_module ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 87
        
        """
        return _solver.f90wrap_setupdt__get__interception_module(self._handle)
    
    @interception_module.setter
    def interception_module(self, interception_module):
        _solver.f90wrap_setupdt__set__interception_module(self._handle, \
            interception_module)
    
    @property
    def production_module(self):
        """
        Element production_module ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 88
        
        """
        return _solver.f90wrap_setupdt__get__production_module(self._handle)
    
    @production_module.setter
    def production_module(self, production_module):
        _solver.f90wrap_setupdt__set__production_module(self._handle, production_module)
    
    @property
    def transfer_module(self):
        """
        Element transfer_module ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 89
        
        """
        return _solver.f90wrap_setupdt__get__transfer_module(self._handle)
    
    @transfer_module.setter
    def transfer_module(self, transfer_module):
        _solver.f90wrap_setupdt__set__transfer_module(self._handle, transfer_module)
    
    @property
    def exchange_module(self):
        """
        Element exchange_module ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 90
        
        """
        return _solver.f90wrap_setupdt__get__exchange_module(self._handle)
    
    @exchange_module.setter
    def exchange_module(self, exchange_module):
        _solver.f90wrap_setupdt__set__exchange_module(self._handle, exchange_module)
    
    @property
    def routing_module(self):
        """
        Element routing_module ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 91
        
        """
        return _solver.f90wrap_setupdt__get__routing_module(self._handle)
    
    @routing_module.setter
    def routing_module(self, routing_module):
        _solver.f90wrap_setupdt__set__routing_module(self._handle, routing_module)
    
    @property
    def save_qsim_domain(self):
        """
        Element save_qsim_domain ftype=logical pytype=bool
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 92
        
        """
        return _solver.f90wrap_setupdt__get__save_qsim_domain(self._handle)
    
    @save_qsim_domain.setter
    def save_qsim_domain(self, save_qsim_domain):
        _solver.f90wrap_setupdt__set__save_qsim_domain(self._handle, save_qsim_domain)
    
    @property
    def save_net_prcp_domain(self):
        """
        Element save_net_prcp_domain ftype=logical pytype=bool
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 93
        
        </> Private
        """
        return _solver.f90wrap_setupdt__get__save_net_prcp_domain(self._handle)
    
    @save_net_prcp_domain.setter
    def save_net_prcp_domain(self, save_net_prcp_domain):
        _solver.f90wrap_setupdt__set__save_net_prcp_domain(self._handle, \
            save_net_prcp_domain)
    
    @property
    def _ntime_step(self):
        """
        Element ntime_step ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 95
        
        """
        return _solver.f90wrap_setupdt__get__ntime_step(self._handle)
    
    @_ntime_step.setter
    def _ntime_step(self, ntime_step):
        _solver.f90wrap_setupdt__set__ntime_step(self._handle, ntime_step)
    
    @property
    def _nd(self):
        """
        Element nd ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 96
        
        """
        return _solver.f90wrap_setupdt__get__nd(self._handle)
    
    @_nd.setter
    def _nd(self, nd):
        _solver.f90wrap_setupdt__set__nd(self._handle, nd)
    
    @property
    def _algorithm(self):
        """
        Element algorithm ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 97
        
        """
        return _solver.f90wrap_setupdt__get__algorithm(self._handle)
    
    @_algorithm.setter
    def _algorithm(self, algorithm):
        _solver.f90wrap_setupdt__set__algorithm(self._handle, algorithm)
    
    @property
    def _jobs_fun(self):
        """
        Element jobs_fun ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 98
        
        """
        return _solver.f90wrap_setupdt__get__jobs_fun(self._handle)
    
    @_jobs_fun.setter
    def _jobs_fun(self, jobs_fun):
        _solver.f90wrap_setupdt__set__jobs_fun(self._handle, jobs_fun)
    
    @property
    def _jreg_fun(self):
        """
        Element jreg_fun ftype=character(lchar) pytype=str
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 99
        
        """
        return _solver.f90wrap_setupdt__get__jreg_fun(self._handle)
    
    @_jreg_fun.setter
    def _jreg_fun(self, jreg_fun):
        _solver.f90wrap_setupdt__set__jreg_fun(self._handle, jreg_fun)
    
    @property
    def _wjreg(self):
        """
        Element wjreg ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 100
        
        """
        return _solver.f90wrap_setupdt__get__wjreg(self._handle)
    
    @_wjreg.setter
    def _wjreg(self, wjreg):
        _solver.f90wrap_setupdt__set__wjreg(self._handle, wjreg)
    
    @property
    def _optim_start_step(self):
        """
        Element optim_start_step ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 101
        
        """
        return _solver.f90wrap_setupdt__get__optim_start_step(self._handle)
    
    @_optim_start_step.setter
    def _optim_start_step(self, optim_start_step):
        _solver.f90wrap_setupdt__set__optim_start_step(self._handle, optim_start_step)
    
    @property
    def _optim_parameters(self):
        """
        Element optim_parameters ftype=integer pytype=int
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 102
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_setupdt__array__optim_parameters(self._handle)
        if array_handle in self._arrays:
            optim_parameters = self._arrays[array_handle]
        else:
            optim_parameters = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_setupdt__array__optim_parameters)
            self._arrays[array_handle] = optim_parameters
        return optim_parameters
    
    @_optim_parameters.setter
    def _optim_parameters(self, optim_parameters):
        self._optim_parameters[...] = optim_parameters
    
    @property
    def _optim_states(self):
        """
        Element optim_states ftype=integer pytype=int
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 103
        
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_setupdt__array__optim_states(self._handle)
        if array_handle in self._arrays:
            optim_states = self._arrays[array_handle]
        else:
            optim_states = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_setupdt__array__optim_states)
            self._arrays[array_handle] = optim_states
        return optim_states
    
    @_optim_states.setter
    def _optim_states(self, optim_states):
        self._optim_states[...] = optim_states
    
    @property
    def _lb_parameters(self):
        """
        Element lb_parameters ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 112
        
        lr
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_setupdt__array__lb_parameters(self._handle)
        if array_handle in self._arrays:
            lb_parameters = self._arrays[array_handle]
        else:
            lb_parameters = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_setupdt__array__lb_parameters)
            self._arrays[array_handle] = lb_parameters
        return lb_parameters
    
    @_lb_parameters.setter
    def _lb_parameters(self, lb_parameters):
        self._lb_parameters[...] = lb_parameters
    
    @property
    def _ub_parameters(self):
        """
        Element ub_parameters ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 121
        
        lr
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_setupdt__array__ub_parameters(self._handle)
        if array_handle in self._arrays:
            ub_parameters = self._arrays[array_handle]
        else:
            ub_parameters = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_setupdt__array__ub_parameters)
            self._arrays[array_handle] = ub_parameters
        return ub_parameters
    
    @_ub_parameters.setter
    def _ub_parameters(self, ub_parameters):
        self._ub_parameters[...] = ub_parameters
    
    @property
    def _lb_states(self):
        """
        Element lb_states ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 127
        
        hlr
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_setupdt__array__lb_states(self._handle)
        if array_handle in self._arrays:
            lb_states = self._arrays[array_handle]
        else:
            lb_states = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_setupdt__array__lb_states)
            self._arrays[array_handle] = lb_states
        return lb_states
    
    @_lb_states.setter
    def _lb_states(self, lb_states):
        self._lb_states[...] = lb_states
    
    @property
    def _ub_states(self):
        """
        Element ub_states ftype=real(sp) pytype=float
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 133
        
        hlr
        """
        array_ndim, array_type, array_shape, array_handle = \
            _solver.f90wrap_setupdt__array__ub_states(self._handle)
        if array_handle in self._arrays:
            ub_states = self._arrays[array_handle]
        else:
            ub_states = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    self._handle,
                                    _solver.f90wrap_setupdt__array__ub_states)
            self._arrays[array_handle] = ub_states
        return ub_states
    
    @_ub_states.setter
    def _ub_states(self, ub_states):
        self._ub_states[...] = ub_states
    
    @property
    def _maxiter(self):
        """
        Element maxiter ftype=integer  pytype=int
        
        
        Defined at smash/solver/module/mwd_setup.f90 line 134
        
        """
        return _solver.f90wrap_setupdt__get__maxiter(self._handle)
    
    @_maxiter.setter
    def _maxiter(self, maxiter):
        _solver.f90wrap_setupdt__set__maxiter(self._handle, maxiter)
    
    def __str__(self):
        ret = ['<setupdt>{\n']
        ret.append('    dt : ')
        ret.append(repr(self.dt))
        ret.append(',\n    start_time : ')
        ret.append(repr(self.start_time))
        ret.append(',\n    sparse_storage : ')
        ret.append(repr(self.sparse_storage))
        ret.append(',\n    read_qobs : ')
        ret.append(repr(self.read_qobs))
        ret.append(',\n    qobs_directory : ')
        ret.append(repr(self.qobs_directory))
        ret.append(',\n    read_prcp : ')
        ret.append(repr(self.read_prcp))
        ret.append(',\n    prcp_format : ')
        ret.append(repr(self.prcp_format))
        ret.append(',\n    prcp_conversion_factor : ')
        ret.append(repr(self.prcp_conversion_factor))
        ret.append(',\n    prcp_directory : ')
        ret.append(repr(self.prcp_directory))
        ret.append(',\n    read_pet : ')
        ret.append(repr(self.read_pet))
        ret.append(',\n    pet_format : ')
        ret.append(repr(self.pet_format))
        ret.append(',\n    pet_conversion_factor : ')
        ret.append(repr(self.pet_conversion_factor))
        ret.append(',\n    pet_directory : ')
        ret.append(repr(self.pet_directory))
        ret.append(',\n    daily_interannual_pet : ')
        ret.append(repr(self.daily_interannual_pet))
        ret.append(',\n    mean_forcing : ')
        ret.append(repr(self.mean_forcing))
        ret.append(',\n    read_descriptor : ')
        ret.append(repr(self.read_descriptor))
        ret.append(',\n    descriptor_format : ')
        ret.append(repr(self.descriptor_format))
        ret.append(',\n    descriptor_directory : ')
        ret.append(repr(self.descriptor_directory))
        ret.append(',\n    descriptor_name : ')
        ret.append(repr(self.descriptor_name))
        ret.append(',\n    interception_module : ')
        ret.append(repr(self.interception_module))
        ret.append(',\n    production_module : ')
        ret.append(repr(self.production_module))
        ret.append(',\n    transfer_module : ')
        ret.append(repr(self.transfer_module))
        ret.append(',\n    exchange_module : ')
        ret.append(repr(self.exchange_module))
        ret.append(',\n    routing_module : ')
        ret.append(repr(self.routing_module))
        ret.append(',\n    save_qsim_domain : ')
        ret.append(repr(self.save_qsim_domain))
        ret.append(',\n    save_net_prcp_domain : ')
        ret.append(repr(self.save_net_prcp_domain))
        ret.append('}')
        return ''.join(ret)

    def copy(self):
        return copy_setup(self)
    
    _dt_array_initialisers = []
    


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module "mwd_setup".')

for func in _dt_array_initialisers:
    func()
