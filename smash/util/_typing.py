from __future__ import annotations

import numpy as np
from os import PathLike

from typing import Union, Tuple, Any
from numpy.typing import NDArray


Numeric = Union[float, int, np.number]

AlphaNumeric = Union[str, Numeric]

ListLike = Union[list, tuple, NDArray]

FilePath = Union[str, "PathLike[str]"]

AnyTuple = Tuple[Any]
