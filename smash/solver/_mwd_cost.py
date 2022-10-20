"""
Module mwd_cost


Defined at smash/solver/module/mwd_cost.f90 lines 16-346

This module `mwd_cost` encapsulates all SMASH cost(type, subroutines, functions)
This module is wrapped and differentiated.

contains

[1]  compute_jobs
[2]  compute_jreg
[3]  compute_cost
[4]  nse
[5]  kge_components
[6]  kge
[7]  se
[8]  rmse
[9]  logarithmique
[10] reg_prior
"""
from __future__ import print_function, absolute_import, division
from smash.solver import _solver
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

def compute_jobs(self, mesh, input_data, output):
    """
    jobs = compute_jobs(self, mesh, input_data, output)
    
    
    Defined at smash/solver/module/mwd_cost.f90 lines 27-85
    
    Parameters
    ----------
    setup : Setupdt
    mesh : Meshdt
    input_data : Input_Datadt
    output : Outputdt
    
    Returns
    -------
    jobs : float
    
    Notes
    -----
    
    Jobs computation subroutine
    
    Given SetupDT, MeshDT, Input_DataDT, OutputDT,
    it returns the result of Jobs computation
    
    Jobs = f(Q*,Q)
    
    See Also
    --------
    nse
    kge
    se
    rmse
    logarithmique
    only: sp, dp, lchar, np, ns
    only: SetupDT
    only: MeshDT
    only: Input_DataDT
    only: ParametersDT
    only: StatesDT
    only: OutputDT
    """
    jobs = _solver.f90wrap_compute_jobs(setup=self._handle, mesh=mesh._handle, \
        input_data=input_data._handle, output=output._handle)
    return jobs

def compute_jreg(self, mesh, parameters, parameters_bgd, states, states_bgd, \
    jreg):
    """
    compute_jreg(self, mesh, parameters, parameters_bgd, states, states_bgd, jreg)
    
    
    Defined at smash/solver/module/mwd_cost.f90 lines 88-124
    
    Parameters
    ----------
    setup : Setupdt
    mesh : Meshdt
    parameters : Parametersdt
    parameters_bgd : Parametersdt
    states : Statesdt
    states_bgd : Statesdt
    jreg : float
    
    Notes
    -----
    
    Jreg computation subroutine
    
    Given SetupDT, MeshDT, ParametersDT, ParametersDT_bgd, StatesDT, STatesDT_bgd,
    it returns the result of Jreg computation
    
    Jreg = f(theta_bgd,theta)
    
    See Also
    --------
    reg_prior
    Normalize prior between parameters and states
    WIP
    """
    _solver.f90wrap_compute_jreg(setup=self._handle, mesh=mesh._handle, \
        parameters=parameters._handle, parameters_bgd=parameters_bgd._handle, \
        states=states._handle, states_bgd=states_bgd._handle, jreg=jreg)

def compute_cost(self, mesh, input_data, parameters, parameters_bgd, states, \
    states_bgd, output, cost):
    """
    compute_cost(self, mesh, input_data, parameters, parameters_bgd, states, \
        states_bgd, output, cost)
    
    
    Defined at smash/solver/module/mwd_cost.f90 lines 126-158
    
    Parameters
    ----------
    setup : Setupdt
    mesh : Meshdt
    input_data : Input_Datadt
    parameters : Parametersdt
    parameters_bgd : Parametersdt
    states : Statesdt
    states_bgd : Statesdt
    output : Outputdt
    cost : float
    
    Notes
    -----
    
    cost computation subroutine
    
    Given SetupDT, MeshDT, Input_DataDT, ParametersDT, ParametersDT_bgd, StatesDT, \
        STatesDT_bgd, OutputDT
    it returns the result of cost computation
    
    cost = Jobs + wJreg * Jreg
    
    See Also
    --------
    compute_jobs
    compute_jreg
    Only compute in case wjreg > 0
    """
    _solver.f90wrap_compute_cost(setup=self._handle, mesh=mesh._handle, \
        input_data=input_data._handle, parameters=parameters._handle, \
        parameters_bgd=parameters_bgd._handle, states=states._handle, \
        states_bgd=states_bgd._handle, output=output._handle, cost=cost)

def nse(x, y):
    """
    res = nse(x, y)
    
    
    Defined at smash/solver/module/mwd_cost.f90 lines 160-196
    
    Parameters
    ----------
    x : float array
    y : float array
    
    Returns
    -------
    res : float
    
    Notes
    -----
    
    NSE computation function
    
    Given two single precision array(x, y) of dim(1) and size(n),
    it returns the result of NSE computation
    num = sum(x**2) - 2 * sum(x*y) + sum(y**2)
    den = sum(x**2) - n * mean(x) ** 2
    NSE = num / den
    NSE numerator / denominator
    NSE criterion
    """
    res = _solver.f90wrap_nse(x=x, y=y)
    return res

def kge_components(x, y, r, a, b):
    """
    kge_components(x, y, r, a, b)
    
    
    Defined at smash/solver/module/mwd_cost.f90 lines 198-240
    
    Parameters
    ----------
    x : float array
    y : float array
    r : float
    a : float
    b : float
    
    Notes
    -----
    
    KGE components computation subroutine
    
    Given two single precision array(x, y) of dim(1) and size(n),
    it returns KGE components r, a, b
    r = cov(x,y) / std(y) / std(x)
    a = mean(y) / mean(x)
    b = std(y) / std(x)
    """
    _solver.f90wrap_kge_components(x=x, y=y, r=r, a=a, b=b)

def kge(x, y):
    """
    res = kge(x, y)
    
    
    Defined at smash/solver/module/mwd_cost.f90 lines 242-263
    
    Parameters
    ----------
    x : float array
    y : float array
    
    Returns
    -------
    res : float
    
    Notes
    -----
    
    KGE computation function
    
    Given two single precision array(x, y) of dim(1) and size(n),
    it returns the result of KGE computation
    KGE = sqrt((1 - r) ** 2 + (1 - a) ** 2 + (1 - b) ** 2)
    
    See Also
    --------
    kge_components
    """
    res = _solver.f90wrap_kge(x=x, y=y)
    return res

def se(x, y):
    """
    res = se(x, y)
    
    
    Defined at smash/solver/module/mwd_cost.f90 lines 265-283
    
    Parameters
    ----------
    x : float array
    y : float array
    
    Returns
    -------
    res : float
    
    Notes
    -----
    
    Square Error(SE) computation function
    
    Given two single precision array(x, y) of dim(1) and size(n),
    it returns the result of SE computation
    SE = sum((x - y) ** 2)
    """
    res = _solver.f90wrap_se(x=x, y=y)
    return res

def rmse(x, y):
    """
    res = rmse(x, y)
    
    
    Defined at smash/solver/module/mwd_cost.f90 lines 285-308
    
    Parameters
    ----------
    x : float array
    y : float array
    
    Returns
    -------
    res : float
    
    Notes
    -----
    
    Root Mean Square Error(RMSE) computation function
    
    Given two single precision array(x, y) of dim(1) and size(n),
    it returns the result of SE computation
    RMSE = sqrt(SE / n)
    
    See Also
    --------
    se
    """
    res = _solver.f90wrap_rmse(x=x, y=y)
    return res

def logarithmique(x, y):
    """
    res = logarithmique(x, y)
    
    
    Defined at smash/solver/module/mwd_cost.f90 lines 310-328
    
    Parameters
    ----------
    x : float array
    y : float array
    
    Returns
    -------
    res : float
    
    Notes
    -----
    
    Logarithmique(LGRM) computation function
    
    Given two single precision array(x, y) of dim(1) and size(n),
    it returns the result of LGRM computation
    LGRM = sum(x * log(y/x) ** 2)
    """
    res = _solver.f90wrap_logarithmique(x=x, y=y)
    return res

def reg_prior(self, size_mat3, matrix, matrix_bgd):
    """
    res = reg_prior(self, size_mat3, matrix, matrix_bgd)
    
    
    Defined at smash/solver/module/mwd_cost.f90 lines 331-346
    
    Parameters
    ----------
    mesh : Meshdt
    size_mat3 : int
    matrix : float array
    matrix_bgd : float array
    
    Returns
    -------
    res : float
    
    Notes
    -----
    
    Prior regularization(PR) computation function
    
    Given two matrix of dim(3) and size(mesh%nrow, mesh%ncol, size_mat3),
    it returns the result of PR computation. (Square Error between matrix)
    
    PR = sum((mat1 - mat2) ** 2)
    TODO refactorize
    """
    res = _solver.f90wrap_reg_prior(mesh=self._handle, size_mat3=size_mat3, \
        matrix=matrix, matrix_bgd=matrix_bgd)
    return res


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module "mwd_cost".')

for func in _dt_array_initialisers:
    func()
