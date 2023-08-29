from __future__ import annotations

from smash._constant import PRECIPITATION_INDICES

from smash.fcore._mw_prcp_indices import precipitation_indices_computation

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model

__all__ = ["PrecipitationIndices", "precipitation_indices"]


class PrecipitationIndices(dict):
    """
    Represents precipitation indices computation result.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors.

    Attributes
    ----------
    TODO FC: Fill

    See Also
    --------
    TODO FC: Fill

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
                [
                    k.rjust(m) + ": " + repr(v)
                    for k, v in sorted(self.items())
                    if not k.startswith("_")
                ]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def precipitation_indices(
    model: Model,
):
    """
    TODO FC: Fill
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
