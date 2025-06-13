from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smash._constant import PRECIPITATION_INDICES
from smash.fcore._mw_prcp_indices import (
    precipitation_indices_computation as wrap_precipitation_indices_computation,
)

if TYPE_CHECKING:
    from smash.core.model.model import Model

__all__ = ["PrecipitationIndices", "precipitation_indices"]


class PrecipitationIndices:
    """
    Represents precipitation indices computation result.

    Attributes
    ----------
    std : `numpy.ndarray`
        The precipitation spatial standard deviation.
    d1 : `numpy.ndarray`
        The first scaled moment :cite:p:`zocatelli_2011`.
    d2 : `numpy.ndarray`
        The second scaled moment :cite:p:`zocatelli_2011`.
    vg : `numpy.ndarray`
        The vertical gap :cite:p:`emmanuel_2015`.
    hg : `numpy.ndarray`
        The horizontal gap :cite:p:`emmanuel_2015`.

    See Also
    --------
    precipitation_indices : Compute precipitation indices of the Model object.
    """

    def __init__(self, data: dict | None = None):
        if data is None:
            data = {}

        self.__dict__.update(data)

    def __repr__(self):
        dct = self.__dict__

        if dct.keys():
            m = max(map(len, list(dct.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(type(v)) for k, v in sorted(dct.items()) if not k.startswith("_")]
            )
        else:
            return self.__class__.__name__ + "()"

    def to_numpy(self, axis: int = 0):
        """
        Convert the `PrecipitationIndices` object to a `numpy.ndarray`.

        The attribute arrays are stacked along a user-specified axis of the resulting array in alphabetical
        order based on the names of the precipitation indices (``'d1'``, ``'d2'``, ``'hg'``, ``'std'``,
        ``'vg'``).

        Parameters
        ----------
        axis : `int`, default 0
            The axis along which the precipitation arrays will be joined.

        Returns
        -------
        res : `numpy.ndarray`
            It returns the `PrecipitationIndices` object as a `numpy.ndarray`.

        Examples
        --------
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Compute precipitation indices

        >>> prcp_ind = smash.precipitation_indices(model)

        Convert the result to a `numpy.ndarray`

        >>> prcp_ind_np = prcp_ind.to_numpy(axis=-1)
        >>> prcp_ind_np
        array([[[nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan],
                ...,
                [nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan]]], dtype=float32)

        >>> prcp_ind_np.shape
        (3, 1440, 5)

        Access a specific precipitation indice

        >>> prcp_ind_keys = sorted(["std", "d1", "d2", "vg", "hg"])
        >>> ind = prcp_ind_keys.index("d1")
        >>> ind
        0

        >>> prcp_ind_np[..., ind]
        array([[nan, nan, nan, ..., nan, nan, nan],
               [nan, nan, nan, ..., nan, nan, nan],
               [nan, nan, nan, ..., nan, nan, nan]], dtype=float32)
        """

        keys = sorted({"std", "d1", "d2", "vg", "hg"})

        return np.stack([getattr(self, k) for k in keys], axis=axis)


def precipitation_indices(
    model: Model,
) -> PrecipitationIndices:
    # % TODO FC: Add advanced user guide
    """
    Compute precipitation indices of Model.

    5 precipitation indices are calculated for each gauge and each time step:

    - ``'std'`` : the precipitation spatial standard deviation
    - ``'d1'`` : the first scaled moment :cite:p:`zocatelli_2011`
    - ``'d2'`` : the second scaled moment :cite:p:`zocatelli_2011`
    - ``'vg'`` : the vertical gap :cite:p:`emmanuel_2015`
    - ``'hg'`` : the horizontal gap :cite:p:`emmanuel_2015`

    Parameters
    ----------
    model : `Model`
        Primary data structure of the hydrological model `smash`.

    Returns
    -------
    precipitation_indices : `PrecipitationIndices`
        It returns an object containing the results of the precipitation indices computation.

    See Also
    --------
    PrecipitationIndices : Represents precipitation indices computation result.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Compute precipitation indices

    >>> prcp_ind = smash.precipitation_indices(model)
    >>> prcp_ind
    d1: <class 'numpy.ndarray'>
    d2: <class 'numpy.ndarray'>
    hg: <class 'numpy.ndarray'>
    std: <class 'numpy.ndarray'>
    vg: <class 'numpy.ndarray'>

    Each attribute is a `numpy.ndarray` of shape *(ng, ntime_step)* (i.e. number of gauges, number of
    time steps)

    Access a specific precipitation indice

    >>> prcp_ind.d1
    array([[nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan]], dtype=float32)

    .. note::
        NaN value means that there is no precipitation at this specific gauge and time step.

    Access a specific precipitation indice for a single gauge

    >>> ind = np.argwhere(model.mesh.code == "V3524010").item()
    >>> ind
    0

    >>> d1g = prcp_ind.d1[ind, :]
    >>> d1g
    array([nan, nan, nan, ..., nan, nan, nan], dtype=float32)

    Access the time steps where rainfall occured for a specific precipitation indice and a single gauge

    >>> ind = np.argwhere(~np.isnan(d1g))
    >>> ind
    array([[  11],
           [  12],
           ...
           [1410],
           [1411]])

    >>> d1gts = d1g[ind]
    >>> d1gts
    array([[1.2091751 ],
           [0.83805513],
           ...
           [0.89658403],
           [0.48382276]], dtype=float32)
    >>> d1gts[0].item(), d1gts[1].item()
    (1.2091751098632812, 0.8380551338195801)
    """

    # % Initialise result
    prcp_indices = np.zeros(
        shape=(len(PRECIPITATION_INDICES), *model.response_data.q.shape),
        dtype=np.float32,
        order="F",
    )

    # % Call Fortran wrapped subroutine
    wrap_precipitation_indices_computation(model.setup, model.mesh, model._input_data, prcp_indices)

    # % Process results by converting negative values to NaN
    prcp_indices = np.where(prcp_indices < 0, np.nan, prcp_indices)

    return PrecipitationIndices(dict(zip(PRECIPITATION_INDICES, prcp_indices)))
