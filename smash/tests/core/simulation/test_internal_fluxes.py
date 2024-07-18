from __future__ import annotations

import os

import numpy as np
import pytest

import smash


def generic_internal_fluxes(model_structure: list[smash.Model], **kwargs) -> dict:
    res = {}

    ncpu = min(5, max(1, os.cpu_count() - 1))

    for model in model_structure:
        # % There is no snow data for the Cance dataset.
        # % TODO: Add a dataset to test snow module
        if model.setup.snow_module_present:
            continue

        instance, ret = smash.forward_run(
            model,
            common_options={"verbose": False, "ncpu": ncpu},
            return_options={
                "internal_fluxes": True,
            },
        )

        mask = model.mesh.active_cell == 1

        for key, value in ret.internal_fluxes.items():
            internal_flux = value[mask, :].flatten()[::15000]
            res[f"internal_fluxes.{instance.setup.structure}.{key}"] = internal_flux

    return res


def run_internal_fluxes_option(model: smash.Model, custom_options) -> dict:
    instance, res = smash.forward_run(
        model,
        **custom_options,
    )
    return res


def test_run_internal_fluxes():
    res = generic_internal_fluxes(pytest.model_structure)

    for key, value in res.items():
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06, equal_nan=True), key


def test_run_internal_fluxes_shapes():
    ncpu = min(5, max(1, os.cpu_count() - 1))
    custom_options = [
        # % Test custom options
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


def optimize_internal_fluxes_shapes(model: smash.Model, custom_options) -> dict:
    instance, res = smash.optimize(
        model,
        **custom_options,
    )
    return res


def test_optimize_internal_fluxes_shapes():
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
            "optimize_options": {
                "termination_crit": {"maxiter": 1},
            },
            "mapping": "distributed",
        },
    ]
    for i, kwargs in enumerate(custom_options):
        res = optimize_internal_fluxes_shapes(pytest.model, kwargs)
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
