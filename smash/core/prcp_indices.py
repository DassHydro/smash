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
    This class is essentially a subclass of dict with attribute accessors.

    Attributes
    ----------
    std : numpy.ndarray
        The precipitation spatial standard deviation.
    d1 : numpy.ndarray
        The first scaled moment, :cite:p:`zocatelli_2011`
    d2 : numpy.ndarray
        The second scaled moment, :cite:p:`zocatelli_2011`
    vg : numpy.ndarray
        The vertical gap. :cite:p:`emmanuel_2015`

    See Also
    --------
    Model.prcp_indices: Compute precipitations indices of the Model.

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
