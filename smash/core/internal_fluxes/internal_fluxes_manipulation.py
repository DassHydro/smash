from __future__ import annotations

from typing import TYPE_CHECKING


def transform_internal_fluxes_to_dict(internal_fluxes, module_dictonnary, structure_name):
    internal_fluxes = {
        key: internal_fluxes[..., i] for i, key in enumerate(module_dictonnary[structure_name])
    }
    return internal_fluxes
