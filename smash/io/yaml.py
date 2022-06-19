from __future__ import annotations

import yaml


def read_yaml_configuration(path: str) -> dict:

    """
    Load yaml configuration file

    """

    with open(path, "r") as f:

        config = yaml.safe_load(f)
        
        if "default_parameters" in config.keys():
            
            print("IN")
        
        print(config)
