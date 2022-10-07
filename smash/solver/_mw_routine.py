"""
Module mw_routine


Defined at smash/solver/module/mw_routine.f90 lines 3-406

This module `mw_routine` encapsulates all SMASH routine.
This module is wrapped
"""
from __future__ import print_function, absolute_import, division
from smash.solver import _solver
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

def copy_setup(self):
    """
    outdt = copy_setup(self)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 15-23
    
    Parameters
    ----------
    indt : Setupdt
    
    Returns
    -------
    outdt : Setupdt
    
    Notes
    -----
    
    SetupDT copy subroutine
    """
    outdt = _solver.f90wrap_copy_setup(indt=self._handle)
    outdt = f90wrap.runtime.lookup_class("solver.SetupDT").from_handle(outdt, \
        alloc=True)
    return outdt

def copy_mesh(self):
    """
    outdt = copy_mesh(self)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 25-33
    
    Parameters
    ----------
    indt : Meshdt
    
    Returns
    -------
    outdt : Meshdt
    
    Notes
    -----
    
    MeshDT copy subroutine
    """
    outdt = _solver.f90wrap_copy_mesh(indt=self._handle)
    outdt = f90wrap.runtime.lookup_class("solver.MeshDT").from_handle(outdt, \
        alloc=True)
    return outdt

def copy_input_data(self):
    """
    outdt = copy_input_data(self)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 35-43
    
    Parameters
    ----------
    indt : Input_Datadt
    
    Returns
    -------
    outdt : Input_Datadt
    
    Notes
    -----
    
    Input_DataDT copy subroutine
    """
    outdt = _solver.f90wrap_copy_input_data(indt=self._handle)
    outdt = f90wrap.runtime.lookup_class("solver.Input_DataDT").from_handle(outdt, \
        alloc=True)
    return outdt

def copy_parameters(self):
    """
    outdt = copy_parameters(self)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 45-53
    
    Parameters
    ----------
    indt : Parametersdt
    
    Returns
    -------
    outdt : Parametersdt
    
    Notes
    -----
    
    ParametersDT copy subroutine
    """
    outdt = _solver.f90wrap_copy_parameters(indt=self._handle)
    outdt = f90wrap.runtime.lookup_class("solver.ParametersDT").from_handle(outdt, \
        alloc=True)
    return outdt

def copy_states(self):
    """
    outdt = copy_states(self)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 55-63
    
    Parameters
    ----------
    indt : Statesdt
    
    Returns
    -------
    outdt : Statesdt
    
    Notes
    -----
    
    StatesDT copy subroutine
    """
    outdt = _solver.f90wrap_copy_states(indt=self._handle)
    outdt = f90wrap.runtime.lookup_class("solver.StatesDT").from_handle(outdt, \
        alloc=True)
    return outdt

def copy_output(self):
    """
    outdt = copy_output(self)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 65-73
    
    Parameters
    ----------
    indt : Outputdt
    
    Returns
    -------
    outdt : Outputdt
    
    Notes
    -----
    
    OutputDT copy subroutine
    """
    outdt = _solver.f90wrap_copy_output(indt=self._handle)
    outdt = f90wrap.runtime.lookup_class("solver.OutputDT").from_handle(outdt, \
        alloc=True)
    return outdt

def compute_rowcol_to_ind_sparse(self):
    """
    compute_rowcol_to_ind_sparse(self)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 75-97
    
    Parameters
    ----------
    mesh : Meshdt
    
    Notes
    -----
    
    (row, col) indices to(k) sparse indices subroutine
    Given MeshDT,
    it saves the link between(row, col) matrix to(k) vector indice
    in MeshDT%rowcol_to_sparse_indice
    """
    _solver.f90wrap_compute_rowcol_to_ind_sparse(mesh=self._handle)

def mask_upstream_cells(row, col, mesh, mask):
    """
    mask_upstream_cells(row, col, mesh, mask)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 99-127
    
    Parameters
    ----------
    row : int
    col : int
    mesh : Meshdt
    mask : bool array
    
    Notes
    -----
    
    Masked upstream cells subroutine
    Given(row, col) indices, MeshDT,
    it returns a mask where True values are upstream cells of
    the given row, col indices.
    """
    _solver.f90wrap_mask_upstream_cells(row=row, col=col, mesh=mesh._handle, \
        mask=mask)

def sparse_matrix_to_vector_r(self, matrix, vector):
    """
    sparse_matrix_to_vector_r(self, matrix, vector)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 129-158
    
    Parameters
    ----------
    mesh : Meshdt
    matrix : float array
    vector : float array
    
    Notes
    -----
    
    Sparse matrix to vector subroutine
    Given MeshDT, a single precision matrix of dim(2) and size(mesh%nrow, \
        mesh%ncol),
    it returns a single precision vector of dim(1) and size(mesh%nac)
    Flatten rule follow mesh%rowcol_to_ind_sparse
    
    See Also
    --------
    compute compute_rowcol_to_ind_sparse
    """
    _solver.f90wrap_sparse_matrix_to_vector_r(mesh=self._handle, matrix=matrix, \
        vector=vector)

def sparse_matrix_to_vector_i(self, matrix, vector):
    """
    sparse_matrix_to_vector_i(self, matrix, vector)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 160-190
    
    Parameters
    ----------
    mesh : Meshdt
    matrix : int array
    vector : int array
    
    Notes
    -----
    
    Sparse matrix to vector subroutine
    Given MeshDT, an integer matrix of dim(2) and size(mesh%nrow, mesh%ncol),
    it returns an integer vector of dim(1) and size(mesh%nac)
    Flatten rule follow mesh%rowcol_to_ind_sparse
    
    See Also
    --------
    compute compute_rowcol_to_ind_sparse
    """
    _solver.f90wrap_sparse_matrix_to_vector_i(mesh=self._handle, matrix=matrix, \
        vector=vector)

def sparse_vector_to_matrix_r(self, vector, matrix, na_value=None):
    """
    sparse_vector_to_matrix_r(self, vector, matrix[, na_value])
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 192-230
    
    Parameters
    ----------
    mesh : Meshdt
    vector : float array
    matrix : float array
    na_value : float
    
    Notes
    -----
    
    Sparse vector to matrix subroutine
    Given MeshDT, a single precision vector of dim(1) and size(mesh%nac),
    optionnaly a single precesion no-data value,
    it returns a single precision matrix of dim(2) and size(mesh%nrow, mesh%ncol)
    Unflatten rule follow mesh%rowcol_to_ind_sparse
    
    See Also
    --------
    compute compute_rowcol_to_ind_sparse
    """
    _solver.f90wrap_sparse_vector_to_matrix_r(mesh=self._handle, vector=vector, \
        matrix=matrix, na_value=na_value)

def sparse_vector_to_matrix_i(self, vector, matrix, na_value=None):
    """
    sparse_vector_to_matrix_i(self, vector, matrix[, na_value])
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 232-269
    
    Parameters
    ----------
    mesh : Meshdt
    vector : int array
    matrix : int array
    na_value : int
    
    Notes
    -----
    
    Sparse vector to matrix subroutine
    Given MeshDT, an integer vector of dim(1) and size(mesh%nac),
    optionnaly an integer no-data value,
    it returns an integer matrix of dim(2) and size(mesh%nrow, mesh%ncol)
    Unflatten rule follow mesh%rowcol_to_ind_sparse
    
    See Also
    --------
    compute compute_rowcol_to_ind_sparse
    """
    _solver.f90wrap_sparse_vector_to_matrix_i(mesh=self._handle, vector=vector, \
        matrix=matrix, na_value=na_value)

def compute_mean_forcing(self, mesh, input_data):
    """
    compute_mean_forcing(self, mesh, input_data)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 271-307
    
    Parameters
    ----------
    setup : Setupdt
    mesh : Meshdt
    input_data : Input_Datadt
    
    Notes
    -----
    
    Mean forcing computation subroutine
    Given SetupDT, MeshDT, Input_DataDT,
    it saves in Input_Data%mean_prcp, Input_DataDT%mean_pet the
    spatial average by catchment of precipitation and evapotranspiration
    for each time step.
    """
    _solver.f90wrap_compute_mean_forcing(setup=self._handle, mesh=mesh._handle, \
        input_data=input_data._handle)

def compute_prcp_indice(self, mesh, input_data):
    """
    compute_prcp_indice(self, mesh, input_data)
    
    
    Defined at smash/solver/module/mw_routine.f90 lines 309-406
    
    Parameters
    ----------
    setup : Setupdt
    mesh : Meshdt
    input_data : Input_Datadt
    
    Notes
    -----
    
    Prcp indice computation subroutine
    Given SetupDT, MeshDT, Input_DataDT,
    it saves in Input_Data%prcp_indice several precipitation indices.
    DOI: doi:10.5194/hess-15-3767-2011(Zoccatelli et al., 2011)
    DOI: https://doi.org/10.1016/j.hydrol.2015.04.058(Emmanuel et al., 2015)
    tmp needed cause array(1, n)
    Do not compute indices if there is no precipitation
    """
    _solver.f90wrap_compute_prcp_indice(setup=self._handle, mesh=mesh._handle, \
        input_data=input_data._handle)


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module "mw_routine".')

for func in _dt_array_initialisers:
    func()
