from __future__ import annotations

from smash.solver._mw_forcing_statistic import compute_prcp_indices

from smash.core._constant import PRCP_INDICES

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import numpy as np


class PrcpIndicesResult(dict):
    """
    Represents the precipitation indices result.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors and an additional method `PrcpIndicesResult.to_numpy`.

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

    See Also
    --------
    Model.prcp_indices: Compute precipitations indices of the Model.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    >>> prcpind = model.prcp_indices()

    Convert the result to a numpy.ndarray:

    >>> prcpind_tonumpy = prcpind.to_numpy()
    >>> prcpind_tonumpy
    array([[[nan, nan, nan, nan],
            [nan, nan, nan, nan],
            [nan, nan, nan, nan],
            ...,
            [nan, nan, nan, nan],
            [nan, nan, nan, nan],
            [nan, nan, nan, nan]]], dtype=float32)

    >>> prcpind_tonumpy.shape
    (3, 1440, 4)

    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(v) for k, v in sorted(self.items())]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())

    def to_numpy(self, axis=0):
        """
        Convert the `PrcpIndicesResult` object to a numpy.ndarray.

        The attribute arrays are stacked along a user-specified axis of the resulting array in alphabetical order
        based on the names of the precipitation indices.

        Parameters
        ----------
        axis : int, default 0
            The axis along which the precipitation arrays will be joined.

        Returns
        -------
        res : numpy.ndarray
            The `PrcpIndicesResult` object as a numpy.ndarray.

        """
        keys = sorted({"std", "d1", "d2", "vg"})

        return np.stack([self[k] for k in keys], axis=axis)


def _prcp_indices(instance: Model) -> PrcpIndicesResult:
    prcp_indices = np.zeros(
        shape=(len(PRCP_INDICES), instance.mesh.ng, instance.setup._ntime_step),
        dtype=np.float32,
        order="F",
    ) - np.float32(1)

    compute_prcp_indices(
        instance.setup, instance.mesh, instance.input_data, prcp_indices
    )

    prcp_indices = np.where(prcp_indices < 0, np.nan, prcp_indices)

    return PrcpIndicesResult(dict(zip(PRCP_INDICES, prcp_indices)))
