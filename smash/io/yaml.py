from __future__ import annotations

import yaml
import numpy as np

from smash.core.common import SMASH_PARAMETERS, SMASH_DEFAULT_PARAMETERS

def read_yaml_configuration(path: str) -> dict:

    """
    Load yaml configuration file

    """

    with open(path, "r") as f:

        config = yaml.safe_load(f)
        
        if "default_parameters" in config.keys():
            
            tmp = dict(zip(SMASH_PARAMETERS, SMASH_DEFAULT_PARAMETERS))
            
            for key, value in config["default_parameters"].items():
                
                tmp.update({key: value})
            
            config["default_parameters"] = list(tmp.values())
            
    return config
