from __future__ import annotations

import numpy as np

SMASH_PARAMETERS = np.array(["cp", "cft", "lr"], dtype="U")
SMASH_DEFAULT_PARAMETERS = np.array([200, 500, 5], dtype=np.float32)
RATIO_PET_HOURLY = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.035,
        0.062,
        0.079,
        0.097,
        0.11,
        0.117,
        0.117,
        0.11,
        0.097,
        0.079,
        0.062,
        0.035,
        0,
        0,
        0,
        0,
        0,
    ], dtype=np.float32
)
