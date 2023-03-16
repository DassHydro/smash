from __future__ import annotations

import smash

from smash.dataset.load import DATASET_NAME

import pytest


def test_load_dataset(**kwargs):
    # % Check dataset loading
    for name in DATASET_NAME:
        try:
            smash.load_dataset(name)
        except:
            pytest.fail(f"load_dataset.{name}")
