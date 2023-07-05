from __future__ import annotations

from smash._constant import NET_OPTIMIZER, LAYER_NAME


def _standardize_layer(layer: str):
    if isinstance(layer, str):
        layer = layer.lower()

        if layer in LAYER_NAME:
            return layer

        else:
            raise ValueError(f"Unknown layer type '{layer}'. Choices: {LAYER_NAME}")

    else:
        raise TypeError(f"layer argument must be str")


def _standardize_optimizer(optimizer: str):
    if isinstance(optimizer, str):
        optimizer = optimizer.lower()

        if optimizer in NET_OPTIMIZER:
            return optimizer

        else:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. Choices: {NET_OPTIMIZER}"
            )

    else:
        raise TypeError(f"optimizer argument must be str")
