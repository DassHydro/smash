from __future__ import annotations

from smash.factory.mesh._tools import _get_transform

import numpy as np
import warnings


def _standardize_gauge(ds_flwdir, x, y, area, code):
    x = np.array(x, dtype=np.float32, ndmin=1)
    y = np.array(y, dtype=np.float32, ndmin=1)
    area = np.array(area, dtype=np.float32, ndmin=1)

    if (x.size != y.size) or (y.size != area.size):
        raise ValueError(
            f"Inconsistent size for 'x' ({x.size}), 'y' ({y.size}) and 'area' ({area.size})"
        )

    xmin, xmax, xres, ymin, ymax, yres = _get_transform(ds_flwdir)

    if np.any((x < xmin) | (x > xmax)):
        raise ValueError(f"'x' {x} value(s) out of flow directions bounds {xmin, xmax}")

    if np.any((y < ymin) | (y > ymax)):
        raise ValueError(f"'y' {y} value(s) out of flow directions bounds {ymin, ymax}")

    if np.any(area < 0):
        raise ValueError(f"Negative 'area' value(s) {area}")

    # % Setting _c0, _c1, ... _cN-1 as default codes
    if code is None:
        code = np.array([f"_c{i}" for i in range(x.size)])

    elif isinstance(code, (str, list)):
        code = np.array(code, ndmin=1)

        # % Only check x (y and area already check above)
        if code.size != x.size:
            raise ValueError(
                f"Inconsistent size for 'code' ({code.size}) and 'x' ({x.size})"
            )

    return x, y, area, code


def _standardize_bbox(ds_flwdir, bbox):
    # % Bounding Box (xmin, xmax, ymin, ymax)

    if isinstance(bbox, (list, tuple, set)):
        bbox = list(bbox)

        if len(bbox) != 4:
            raise ValueError(f"'bbox argument must be of size 4 ({len(bbox)})")

        if bbox[0] > bbox[1]:
            raise ValueError(
                f"'bbox' xmin ({bbox[0]}) is greater than xmax ({bbox[1]})"
            )

        if bbox[2] > bbox[3]:
            raise ValueError(
                f"'bbox' ymin ({bbox[2]}) is greater than ymax ({bbox[3]})"
            )

        xmin, xmax, xres, ymin, ymax, yres = _get_transform(ds_flwdir)

        if bbox[0] < xmin:
            warnings.warn(
                f"'bbox' xmin ({bbox[0]}) is out of flow directions bound ({xmin}). 'bbox' is update according to flow directions bound"
            )

            bbox[0] = xmin

        if bbox[1] > xmax:
            warnings.warn(
                f"'bbox' xmax ({bbox[1]}) is out of flow directions bound ({xmax}). 'bbox' is update according to flow directions bound"
            )

            bbox[1] = xmax

        if bbox[2] < ymin:
            warnings.warn(
                f"'bbox' ymin ({bbox[2]}) is out of flow directions bound ({ymin}). 'bbox' is update according to flow directions bound"
            )

            bbox[2] = ymin

        if bbox[3] > ymax:
            warnings.warn(
                f"'bbox' ymax ({bbox[3]}) is out of flow directions bound ({ymax}). 'bbox' is update according to flow directions bound"
            )

            bbox[3] = ymax

    else:
        raise TypeError("'bbox' argument must be list-like object")

    return bbox
