from __future__ import annotations


def _standardize_event_seg_options(options: dict | None) -> dict:
    if options is None:
        options = {}

    return options
