import numpy as np

import pytest


def test_event_seg():

    arr = pytest.model.event_segmentation().to_numpy()

    arr = arr.astype("S")

    assert np.array_equal(arr, pytest.baseline["event_seg"][:])
