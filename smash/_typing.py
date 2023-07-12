from __future__ import annotations

from typing import Union, List, Tuple, Any

import numpy as np
from os import PathLike


Numeric = Union[float, int]

AlphaNumeric = Union[str, Numeric]

ListLike = Union[List, Tuple]

FilePath = Union[str, "PathLike[str]"]

AnyTuple = Tuple[Any]
