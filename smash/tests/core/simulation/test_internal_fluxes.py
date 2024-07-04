from __future__ import annotations

import os

import pytest

import smash


def run_internal_fluxes_option(model: smash.Model, custom_options) -> dict:
    instance, res = smash.forward_run(
        model,
        **custom_options,
    )
    return res


def run_qt_option(model: smash.Model) -> dict:
    res = model.forward_run(
        return_options={
            "qt": True,
        },
    )
    return res


def test_run_internal_fluxes_option():
    ncpu = min(5, max(1, os.cpu_count() - 1))
    custom_options = [
        # % Test custom optimize_options
        {
            "cost_options": None,
            "return_options": {
                "internal_fluxes": True,
            },
            "common_options": {
                "ncpu": ncpu,
                "verbose": False,
            },
        },
        {
            "cost_options": None,
            "return_options": {
                "internal_fluxes": True,
                "time_step": "all",
            },
            "common_options": {
                "ncpu": ncpu,
                "verbose": False,
            },
        },
        {
            "cost_options": None,
            "return_options": {
                "internal_fluxes": True,
                "time_step": ["2014-10-15", "2014-10-16"],
            },
            "common_options": {
                "ncpu": ncpu,
                "verbose": False,
            },
        },
    ]

    for i, kwargs in enumerate(custom_options):
        res = run_internal_fluxes_option(pytest.model, kwargs)
        assert res is not None
        internal_fluxes = res.internal_fluxes

        if "time_step" in kwargs["return_options"]:
            if kwargs["return_options"]["time_step"] == "all":
                nmts = pytest.model.setup.ntime_step
            elif type(kwargs["return_options"]["time_step"]) is not list:
                nmts = len(kwargs["return_options"]["time_step"].split())
            else:
                nmts = len(kwargs["return_options"]["time_step"])
        else:
            nmts = pytest.model.setup.ntime_step

        for k in internal_fluxes:
            assert internal_fluxes[k].shape == (pytest.model.mesh.nrow, pytest.model.mesh.ncol, nmts)


def test_run_qt_option():
    res = run_qt_option(pytest.model)
    assert res.qt.shape == (pytest.model.mesh.nrow, pytest.model.mesh.ncol, pytest.model.setup.ntime_step)
