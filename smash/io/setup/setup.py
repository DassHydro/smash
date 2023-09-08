from __future__ import annotations

import yaml
import os
import errno

__all__ = ["save_setup", "read_setup"]


def save_setup(setup: dict, path: str):
    """
    Save the Model initialization setup dictionary.

    Parameters
    ----------
    setup : dict
        The setup dictionary to be saved to `YAML <https://yaml.org/spec/1.2.2/>`__ file.

    path : str
        The file path. If the path not end with ``.yaml``, the extension is automatically added to the file path.

    See Also
    --------
    read_setup : Read the Model initialization setup dictionary.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_setup, read_setup
    >>> setup, mesh = load_dataset("cance")
    >>> setup
    {'structure': 'gr4-lr', 'dt': 3600, 'start_time': '2014-09-15 00:00', ...}

    Save setup:

    >>> save_setup(setup, "setup.yaml")

    Read setup (the reloaded setup keys will be alphabetically sorted):

    >>> setup_rld = read_setup("setup.yaml")
    >>> setup_rld
    {'daily_interannual_pet': True, 'descriptor_name': ['slope', 'dd'], ...}
    """

    if not path.endswith(".yaml"):
        path = path + ".yaml"

    with open(path, "w") as f:
        yaml.dump(setup, f, default_flow_style=False)


def read_setup(path: str) -> dict:
    """
    Read the Model initialization setup dictionary.

    Parameters
    ----------
    path : str
        The file path.

    Returns
    -------
    setup : dict
        A setup dictionary loaded from YAML file.

    See Also
    --------
    save_setup : Save the Model initialization setup dictionary.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_setup, read_setup
    >>> setup, mesh = load_dataset("cance")
    >>> setup
    {'structure': 'gr4-lr', 'dt': 3600, 'start_time': '2014-09-15 00:00', ...}

    Save setup:

    >>> save_setup(setup, "setup.yaml")

    Read setup (the reloaded setup keys will be alphabetically sorted):

    >>> setup_rld = read_setup("setup.yaml")
    >>> setup_rld
    {'daily_interannual_pet': True, 'descriptor_name': ['slope', 'dd'], ...}
    """

    if os.path.isfile(path):
        with open(path, "r") as f:
            setup = yaml.safe_load(f)

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    return setup
