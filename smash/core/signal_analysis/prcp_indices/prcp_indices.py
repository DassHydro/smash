from __future__ import annotations

from smash._constant import PRECIPITATION_INDICES

from smash.fcore._mw_prcp_indices import precipitation_indices_computation

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model

__all__ = ["PrecipitationIndices", "precipitation_indices"]


class PrecipitationIndices:

    """
    Represents precipitation indices computation result.

    Attributes
    ----------
    std : numpy.ndarray
        The precipitation spatial standard deviation.
    d1 : numpy.ndarray
        The first scaled moment :cite:p:`zocatelli_2011`.
    d2 : numpy.ndarray
        The second scaled moment :cite:p:`zocatelli_2011`.
    vg : numpy.ndarray
        The vertical gap :cite:p:`emmanuel_2015`.
    hg : numpy.ndarray
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
                [
                    k.rjust(m) + ": " + repr(type(v))
                    for k, v in sorted(dct.items())
                    if not k.startswith("_")
                ]
            )
        else:
            return self.__class__.__name__ + "()"

    def to_numpy(self, axis=0):
        """
        Convert the `PrecipitationIndices` object to a numpy.ndarray.

        The attribute arrays are stacked along a user-specified axis of the resulting array in alphabetical order
        based on the names of the precipitation indices (d1, d2, hg, std, vg).

        Parameters
        ----------
        axis : int, default 0
            The axis along which the precipitation arrays will be joined.

        Returns
        -------
        res : numpy.ndarray
            The `PrecipitationIndices` object as a numpy.ndarray.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)
        >>> prcpind = smash.precipitation_indices(model)

        Convert the result to a numpy.ndarray:

        >>> prcpind_tonumpy = prcpind.to_numpy(axis=-1)
        >>> prcpind_tonumpy
        array([[[nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan],
                ...,
                [nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan]]], dtype=float32)

        >>> prcpind_tonumpy.shape
        (3, 1440, 5)
        """

        keys = sorted({"std", "d1", "d2", "vg", "hg"})

        return np.stack([getattr(self, k) for k in keys], axis=axis)


def precipitation_indices(
    model: Model,
):
    """
    Compute precipitation indices of the Model object.

    5 precipitation indices are calculated for each gauge and each time step:

    - ``std`` : The precipitation spatial standard deviation.
    - ``d1`` : The first scaled moment, :cite:p:`zocatelli_2011`.
    - ``d2`` : The second scaled moment, :cite:p:`zocatelli_2011`.
    - ``vg`` : The vertical gap :cite:p:`emmanuel_2015`.
    - ``hg`` : The horizontal gap :cite:p:`emmanuel_2015`.

    .. hint::
        See the (TODO: Fill) for more.

    Returns
    -------
    res : PrecipitationIndices
        The precipitation indices results represented as a `PrecipitationIndices` object.

    Examples
    --------
    Examples
    --------
    >>> import smash
    >>> from smash.factory import load_dataset
    >>> import numpy as np
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Compute precipitation indices:

    >>> prcp_ind = smash.precipitation_indices(model)
    >>> prcp_ind
    d1: <class 'numpy.ndarray'>
    d2: <class 'numpy.ndarray'>
    hg: <class 'numpy.ndarray'>
    std: <class 'numpy.ndarray'>
    vg: <class 'numpy.ndarray'>

    Each attribute is a numpy.ndarray of shape (number of gauge, number of time step):

    >>> prcp_ind.d1.shape
    (3, 1440)

    NaN value means that there is no precipitation at this specific gauge and time step. Using numpy.where to find the index where precipitation indices were calculated on the most downstream gauge for the first scaled moment:

    >>> ind = np.argwhere(~np.isnan(prcp_ind.d1[0,:])).squeeze()

    Viewing the first scaled moment on the first time step where rainfall occured on the most downstream gauge:

    >>> prcp_ind.d1[0, ind[0]]
    1.209175

    See Also
    --------
    PrecipitationIndices : Represents precipitation indices computation result.
    """

    # % Initialise result
    prcp_indices = np.zeros(
        shape=(len(PRECIPITATION_INDICES), *model.obs_response.q.shape),
        dtype=np.float32,
        order="F",
    )

    # % Call Fortran wrapped subroutine
    precipitation_indices_computation(
        model.setup, model.mesh, model._input_data, prcp_indices
    )

    # % Process results by converting negative values to NaN
    prcp_indices = np.where(prcp_indices < 0, np.nan, prcp_indices)

    return PrecipitationIndices(dict(zip(PRECIPITATION_INDICES, prcp_indices)))
