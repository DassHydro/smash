from __future__ import annotations

import smash

from smash._constant import DATASET_NAME

import pytest


def test_load_dataset(**kwargs):
    # % Check dataset loading
    for name in DATASET_NAME:
        try:
            smash.factory.load_dataset(name)
        except:
            pytest.fail(f"load_dataset.{name}")
