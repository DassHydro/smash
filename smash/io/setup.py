from __future__ import annotations

import yaml
import os
import errno

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.util._typing import FilePath
    from typing import Any

__all__ = ["save_setup", "read_setup"]


def save_setup(setup: dict[str, Any], path: FilePath):
    """
    Save the Model initialization setup dictionary to YAML.

    Parameters
    ----------
    setup : `dict[str, Any]`
        The setup dictionary to be saved to `YAML <https://yaml.org/spec/1.2.2/>`__ file.

    path : `str`
        The file path. If the path not end with ``.yaml``, the extension is automatically added to the file path.

    See Also
    --------
    read_setup : Read the Model initialization setup dictionary from YAML.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_setup, read_setup
    >>> setup, mesh = load_dataset("cance")
    >>> setup
    {'structure': 'gr4-lr', 'dt': 3600, 'start_time': '2014-09-15 00:00', ...}

    Save setup to YAML

    >>> save_setup(setup, "setup.yaml")
    """

    if not path.endswith(".yaml"):
        path = path + ".yaml"

    with open(path, "w") as f:
        yaml.dump(setup, f, default_flow_style=False)


def read_setup(path: FilePath) -> dict[str, Any]:
    """
    Read the Model initialization setup dictionary from YAML.

    Parameters
    ----------
    path : `str`
        The file path.

    Returns
    -------
    setup : dict[str, Any]
        A setup dictionary loaded from YAML.

    Raises
    ------
    FileNotFoundError:
        If file not found.

    See Also
    --------
    save_setup : Save the Model initialization setup dictionary to YAML.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_setup, read_setup
    >>> setup, mesh = load_dataset("cance")
    >>> setup
    {'structure': 'gr4-lr', 'dt': 3600, 'start_time': '2014-09-15 00:00', ...}

    Save setup to YAML

    >>> save_setup(setup, "setup.yaml")

    Read setup from YAML (the reloaded setup keys will be alphabetically sorted)

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
