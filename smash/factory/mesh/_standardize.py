from __future__ import annotations

import errno
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import rasterio

from smash.factory.mesh._tools import _get_transform

if TYPE_CHECKING:
    from typing import Tuple

    from smash.util._typing import AlphaNumeric, AnyTuple, FilePath, ListLike, Numeric


def _standardize_generate_mesh_flwdir_path(flwdir_path: FilePath) -> str:
    if not isinstance(flwdir_path, (str, os.PathLike)):
        raise TypeError("flwdir_path argument must be of FilePath type (str, PathLike[str])")

    flwdir_path = str(flwdir_path)

    if not os.path.exists(flwdir_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), flwdir_path)

    return flwdir_path


def _standardize_generate_mesh_bbox(flwdir_dataset: rasterio.DatasetReader, bbox: ListLike) -> np.ndarray:
    # % Bounding Box (xmin, xmax, ymin, ymax)

    if not isinstance(bbox, (list, tuple, np.ndarray)):
        raise TypeError("bbox argument must be of ListLike type (List, Tuple, np.ndarray)")

    bbox = np.array(bbox)

    if bbox.size != 4:
        raise ValueError(f"bbox argument must be of size 4 ({bbox.size})")

    if bbox[0] > bbox[1]:
        raise ValueError(f"bbox xmin ({bbox[0]}) is greater than xmax ({bbox[1]})")

    if bbox[2] > bbox[3]:
        raise ValueError(f"bbox ymin ({bbox[2]}) is greater than ymax ({bbox[3]})")

    xmin, xmax, xres, ymin, ymax, yres = _get_transform(flwdir_dataset)

    if bbox[0] < xmin:
        warnings.warn(
            f"bbox xmin ({bbox[0]}) is out of flow directions bound ({xmin}). bbox is update according to "
            f"flow directions bound",
            stacklevel=2,
        )

        bbox[0] = xmin

    if bbox[1] > xmax:
        warnings.warn(
            f"bbox xmax ({bbox[1]}) is out of flow directions bound ({xmax}). bbox is update according to "
            f"flow directions bound",
            stacklevel=2,
        )

        bbox[1] = xmax

    if bbox[2] < ymin:
        warnings.warn(
            f"bbox ymin ({bbox[2]}) is out of flow directions bound ({ymin}). bbox is update according to "
            f"flow directions bound",
            stacklevel=2,
        )

        bbox[2] = ymin

    if bbox[3] > ymax:
        warnings.warn(
            f"bbox ymax ({bbox[3]}) is out of flow directions bound ({ymax}). bbox is update according to "
            f"flow directions bound",
            stacklevel=2,
        )

        bbox[3] = ymax

    # % Pad the bounding box so that the origins overlap
    bbox[0:2] = xmin + np.rint((bbox[0:2] - xmin) / xres) * xres
    bbox[2:4] = ymax - np.rint((ymax - bbox[2:4]) / yres) * yres

    return bbox


def _standardize_generate_mesh_x_y_area(
    flwdir_dataset: rasterio.DatasetReader,
    x: Numeric | ListLike,
    y: Numeric | ListLike,
    area: Numeric | ListLike,
) -> Tuple[np.ndarray]:
    if not isinstance(x, (int, float, list, tuple, np.ndarray)):
        raise TypeError(
            "x argument must be of Numeric type (int, float) or ListLike type (List, Tuple, np.ndarray)"
        )

    if not isinstance(y, (int, float, list, tuple, np.ndarray)):
        raise TypeError(
            "y argument must be of Numeric type (int, float) or ListLike type (List, Tuple, np.ndarray)"
        )

    if not isinstance(area, (int, float, list, tuple, np.ndarray)):
        raise TypeError(
            "area argument must be of Numeric type (int, float) or ListLike type (List, Tuple, np.ndarray)"
        )

    x = np.array(x, dtype=np.float32, ndmin=1)
    y = np.array(y, dtype=np.float32, ndmin=1)
    area = np.array(area, dtype=np.float32, ndmin=1)

    if (x.size != y.size) or (y.size != area.size):
        raise ValueError(f"Inconsistent sizes between x ({x.size}), y ({y.size}) and area ({area.size})")

    xmin, xmax, _, ymin, ymax, _ = _get_transform(flwdir_dataset)

    if np.any((x < xmin) | (x > xmax)):
        raise ValueError(f"x {x} value(s) out of flow directions bounds {xmin, xmax}")

    if np.any((y < ymin) | (y > ymax)):
        raise ValueError(f"y {y} value(s) out of flow directions bounds {ymin, ymax}")

    if np.any(area < 0):
        raise ValueError(f"area {area} value(s) must be positive")

    return x, y, area


def _standardize_generate_mesh_code(x: np.ndarray, code: str | ListLike | None) -> np.ndarray:
    if code is None:
        code = np.array([f"_c{i}" for i in range(x.size)])

    else:
        if not isinstance(code, (str, list, tuple, np.ndarray)):
            raise TypeError("code argument must be a str or ListLike type (List, Tuple, np.ndarray)")

        code = np.array(code, ndmin=1)

        # % Only check x (y and area already check)
        if code.size != x.size:
            raise ValueError(f"Inconsistent size between code ({code.size}) and x ({x.size})")
    return code


def _standardize_generate_mesh_max_depth(max_depth: Numeric) -> int:
    if not isinstance(max_depth, (int, float)):
        raise TypeError("max_depth argument must be of Numeric type (int, float)")

    max_depth = int(max_depth)

    if max_depth < 0:
        raise ValueError(f"max_depth {max_depth} value must be positive")

    return max_depth


def _standardize_generate_mesh_epsg(epsg: AlphaNumeric | None) -> int:
    if epsg is None:
        pass

    else:
        if not isinstance(epsg, (str, int, float)):
            raise TypeError("epsg argument must be of AlphaNumeric type (str, int, float)")

        epsg = int(epsg)

    return epsg


def _standardize_generate_mesh_args(
    flwdir_path: FilePath,
    bbox: ListLike | None,
    x: Numeric | ListLike | None,
    y: Numeric | ListLike | None,
    area: Numeric | ListLike | None,
    code: str | ListLike | None,
    max_depth: Numeric,
    epsg: AlphaNumeric | None,
) -> AnyTuple:
    flwdir_path = _standardize_generate_mesh_flwdir_path(flwdir_path)

    flwdir_dataset = rasterio.open(flwdir_path)

    if x is None and bbox is None:
        raise ValueError("bbox argument or (x, y, area) arguments must be defined")

    if bbox is not None:
        bbox = _standardize_generate_mesh_bbox(flwdir_dataset, bbox)

    else:
        x, y, area = _standardize_generate_mesh_x_y_area(flwdir_dataset, x, y, area)

        code = _standardize_generate_mesh_code(x, code)

    max_depth = _standardize_generate_mesh_max_depth(max_depth)

    epsg = _standardize_generate_mesh_epsg(epsg)

    return (flwdir_dataset, bbox, x, y, area, code, max_depth, epsg)
