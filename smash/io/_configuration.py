from __future__ import annotations

import yaml
import logging


def _read_yaml_configuration(path: str) -> dict:

    """
    Load yaml configuration file

    """

    with open(path, "r") as f:

        return yaml.safe_load(f)
