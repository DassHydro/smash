from __future__ import annotations

from typing import TYPE_CHECKING

from smash._constant import ACTIVATION_FUNCTION, LAYER_NAME

if TYPE_CHECKING:
    from smash.util._typing import AnyTuple


def _standardize_add_layer(layer: str) -> str:
    if isinstance(layer, str):
        layer_standardized = layer.lower().capitalize()

        if layer_standardized in LAYER_NAME:
            layer = layer_standardized

        else:
            raise ValueError(f"Unknown layer type '{layer}'. Choices: {LAYER_NAME}")

    else:
        raise TypeError("layer argument must be str")

    return layer


def _standardize_add_options(layer: str, options: dict) -> dict:
    if isinstance(options, dict):
        if layer == "Activation":
            try:
                option_name = options["name"]

            except KeyError:
                raise ValueError(
                    "Key 'name' in options argument must be specified for the activation function"
                ) from None

            if isinstance(option_name, str):
                activation_function = [name.lower() for name in ACTIVATION_FUNCTION]

                try:
                    ind = activation_function.index(option_name.lower())

                except ValueError:
                    raise ValueError(
                        f"Unknown activation function '{option_name}'. Choices: {ACTIVATION_FUNCTION}"
                    ) from None

                options["name"] = ACTIVATION_FUNCTION[ind]

            else:
                raise TypeError("Key 'name' in options argument must be a str")

        else:
            pass

    else:
        raise TypeError("options argument must be a dictionary")

    return options


def _standardize_add_args(layer: str, options: dict) -> AnyTuple:
    layer = _standardize_add_layer(layer)

    options = _standardize_add_options(layer, options)

    return (layer, options)
