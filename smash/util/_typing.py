from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from os import PathLike

# TODO TYPE HINT: ENH smash typing

Numeric = Union[float, int, np.number]

AlphaNumeric = Union[str, Numeric]

ListLike = Union[list, tuple, NDArray]

FilePath = Union[str, "PathLike[str]"]

AnyTuple = Tuple[Any]
