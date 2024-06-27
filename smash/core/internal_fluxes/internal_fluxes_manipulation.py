from __future__ import annotations

from typing import TYPE_CHECKING

def transform_internal_fluxes_to_dict(internal_fluxes, module_dictonnary, structure_name):
    internal_fluxes = dict(
                zip(module_dictonnary[structure_name], internal_fluxes)
            )
    return internal_fluxes