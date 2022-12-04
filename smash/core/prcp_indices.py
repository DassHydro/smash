from __future__ import annotations

from smash.core._constant import PRCP_INDICES

from smash.solver._mw_forcing_statistic import compute_prcp_indices

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
        The first scaled moment, [1]_
    d2 : numpy.ndarray
        The second scaled moment, [1]_
    vg : numpy.ndarray
        The vertical gap. [2]_

    See Also
    --------
    Model.prcp_indices: Compute precipitations indices of the Model.

    References
    ----------
    .. [1]

        Zoccatelli, D., Borga, M., Viglione, A., Chirico, G. B., and Blöschl, G.:
        Spatial moments of catchment rainfall: rainfall spatial organisation,
        basin morphology, and flood response,
        Hydrol. Earth Syst. Sci., 15, 3767–3783,
        https://doi.org/10.5194/hess-15-3767-2011, 2011.

    .. [2]

        I. Emmanuel, H. Andrieu, E. Leblois, N. Janey, O. Payrastre,
        Influence of rainfall spatial variability on rainfall–runoff modelling:
        Benefit of a simulation approach?,
        Journal of Hydrology,
        https://doi.org/10.1016/j.jhydrol.2015.04.058, 2015

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
