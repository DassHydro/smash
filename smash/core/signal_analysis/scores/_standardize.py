from __future__ import annotations

from smash._constant import EFFICIENCY_METRICS


def _standardize_metric(metric: str):
    if isinstance(metric, str):
        if metric in EFFICIENCY_METRICS:
            return metric.lower()

        else:
            raise ValueError(
                f"Unknown efficiency metric {metric}. Choices: {EFFICIENCY_METRICS}"
            )
    else:
        raise TypeError(f"metric must be str")
