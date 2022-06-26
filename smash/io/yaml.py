from __future__ import annotations

import yaml
import numpy as np

from smash.core.common import (

    SMASH_CONFIGURATION_DICT,
)


def _standardize_configuration_dict(configuration, smash_key, smash_value):

    tmp = dict(zip(smash_value[0], smash_value[1]))

    for key, value in configuration[smash_key].items():

        if key in tmp.keys():

            tmp.update({key: value})

        else:

            raise ValueError(f"Invalid key '{key}' in '{smash_key}'")
            
    return list(tmp.values())


def read_yaml_configuration(path: str) -> dict:

    """
    Load yaml configuration file

    """

    with open(path, "r") as f:

        configuration = yaml.safe_load(f)
        
        for smash_key, smash_value in SMASH_CONFIGURATION_DICT.items():
            
            if smash_key in configuration.keys():
            
                configuration[smash_key] = _standardize_configuration_dict(configuration, smash_key, smash_value)

    return configuration
