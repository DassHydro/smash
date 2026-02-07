from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from smash._constant import DOMAIN, GAUGE_ALIAS

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.util._typing import AnyTuple, ListLike, Numeric


def _standardize_hydrograph_segmentation_peak_quant(peak_quant: Numeric) -> float:
    if isinstance(peak_quant, (int, float)):
        if peak_quant < 0 or peak_quant > 1:
            raise ValueError("peak_quant argument must be between 0 and 1")
        else:
            peak_quant = float(peak_quant)

    else:
        raise TypeError("peak_quant argument must be Numeric (float, int)")

    return peak_quant


def _standardize_hydrograph_segmentation_peak_value(peak_value: Numeric) -> float:
    if isinstance(peak_value, (int, float)):
        if peak_value < 0:
            raise ValueError("peak_value argument must be non-negative")
        else:
            peak_value = float(peak_value)

    else:
        raise TypeError("peak_value argument must be Numeric (float, int)")

    return peak_value


def _standardize_hydrograph_segmentation_max_duration(
    max_duration: Numeric,
) -> float:
    if not isinstance(max_duration, (int, float)):
        raise TypeError("max_duration argument must be of Numeric type (int, float)")

    max_duration = float(max_duration)

    if max_duration <= 0:
        raise ValueError("max_duration argument must be greater than 0")

    return max_duration


def _standardize_hydrograph_segmentation_by(by: str) -> str:
    if isinstance(by, str):
        by_standardized = by.lower()

        if by_standardized in DOMAIN:
            by_standardized = by_standardized[:3]

        else:
            raise ValueError(f"Unknown by argument {by}. Choices: {DOMAIN}")
    else:
        raise TypeError("by argument must be str")

    return by_standardized


def _standardize_hydrograph_segmentation_gauge(
    model: Model,
    by: str,
    gauge: str | ListLike,
) -> np.ndarray:
    if isinstance(gauge, str):
        if gauge == "dws":
            gauge = np.empty(shape=0)

            for i, pos in enumerate(model.mesh.gauge_pos):
                if model.mesh.flwdst[tuple(pos)] == 0:
                    gauge = np.append(gauge, model.mesh.code[i])

        elif gauge == "all":
            gauge = np.array(model.mesh.code, ndmin=1)

        elif gauge in model.mesh.code:
            gauge = np.array(gauge, ndmin=1)

        else:
            raise ValueError(
                f"Unknown alias or gauge code '{gauge}' for gauge cost_options. "
                f"Choices: {GAUGE_ALIAS + list(model.mesh.code)}"
            )

    elif isinstance(gauge, (list, tuple, np.ndarray)):
        gauge_bak = np.array(gauge, ndmin=1)
        gauge = np.empty(shape=0)

        for ggc in gauge_bak:
            if ggc in model.mesh.code:
                gauge = np.append(gauge, ggc)
            else:
                raise ValueError(
                    f"Unknown gauge code '{ggc}' in gauge cost_options. Choices: {list(model.mesh.code)}"
                )

    else:
        raise TypeError("gauge cost_options must be a str or ListLike type (List, Tuple, np.ndarray)")

    # % Check that there is observed/simulated discharge or mean precipitation available for segmentation
    if by == "obs":
        q = model.response_data.q
    elif by == "sim":
        q = model.response.q
    else:  # should be unreachable
        pass

    gauge_bak = np.array(gauge, ndmin=1)
    gauge = np.empty(shape=0)
    for i, ggc in enumerate(model.mesh.code):
        if ggc in gauge_bak:
            if np.all(q[i] < 0):  # check q first
                warnings.warn(
                    f"No {by} discharge available at gauge '{ggc}' for segmentation",
                    stacklevel=2,
                )

            elif np.all(model.atmos_data.mean_prcp[i] < 0):  # check mean_prcp
                warnings.warn(
                    f"No precipitation data available at gauge '{ggc}' for segmentation",
                    stacklevel=2,
                )

            else:
                gauge = np.append(gauge, ggc)

    return gauge


def _standardize_hydrograph_segmentation_args(
    model: Model,
    gauge: str | ListLike,
    peak_quant: Numeric,
    peak_value: Numeric,
    max_duration: Numeric,
    by: str,
) -> AnyTuple:
    peak_quant = _standardize_hydrograph_segmentation_peak_quant(peak_quant)

    peak_value = _standardize_hydrograph_segmentation_peak_value(peak_value)

    max_duration = _standardize_hydrograph_segmentation_max_duration(max_duration)

    by = _standardize_hydrograph_segmentation_by(by)

    gauge = _standardize_hydrograph_segmentation_gauge(model, by, gauge)

    return (gauge, peak_quant, peak_value, max_duration, by)
