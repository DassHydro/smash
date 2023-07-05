from __future__ import annotations

from smash.factory.mesh._get_mesh import _get_mesh_from_xy, _get_mesh_from_bbox

import errno
import os
from osgeo import gdal


__all__ = ["generate_mesh"]


def generate_mesh(
    path: str,
    bbox: list | tuple | None = None,
    x: float | list | tuple | None = None,
    y: float | list | tuple | None = None,
    area: float | list | tuple | None = None,
    code: str | list | tuple | None = None,
    max_depth: int = 1,
    epsg: str | int | None = None,
) -> dict:
    """
    Automatic mesh generation.

    .. hint::
        See the :ref:`User Guide <user_guide.in_depth.automatic_meshing>` for more.

    Parameters
    ----------
    path : str
        Path to the flow directions file. The file will be opened with `gdal <https://gdal.org/api/python/osgeo.gdal.html>`__.

    bbox : sequence or None, default None
        Bounding box with the following convention:

        ``(xmin, xmax, ymin, ymax)``.
        The bounding box values must respect the CRS of the flow directions file.

        .. note::
            If not given, **x**, **y** and **area** must be filled in.

    x : float, sequence or None, default None
        The x-coordinate(s) of the catchment outlet(s) to mesh.
        The **x** value(s) must respect the CRS of the flow directions file.
        The **x** size must be equal to **y** and area.

    y : float, sequence or None, default None
        The y-coordinate(s) of the catchment outlet(s) to mesh.
        The **y** value(s) must respect the CRS of the flow directions file.
        The **y** size must be equal to **x** and **area**.

    area : float, sequence or None, default None
        The area of the catchment(s) to mesh in m².
        The **area** size must be equal to **x** and **y**.

    code : str, sequence or None, default None
        The code of the catchment(s) to mesh.
        The **code** size must be equal to **x**, **y** and **area**.
        In case of bouding box meshing, the **code** argument is not used.

        .. note::
            If not given, the default code is:

            ``['_c0', '_c1', ..., '_cn-1']`` with :math:`n`, the number of gauges.

    max_depth : int, default 1
        The maximum depth accepted by the algorithm to find the catchment outlet.
        A **max_depth** of 1 means that the algorithm will search among the 2-length combinations in:

        ``(row - 1, row, row + 1, col - 1, col, col + 1)``, the coordinates that minimize the relative error between
        the given catchment area and the modeled catchment area calculated from the flow directions file.
        This can be generalized to :math:`n`.

        .. image:: ../../_static/max_depth.png
            :align: center
            :width: 350

    epsg : str, int or None, default None
        The EPSG value of the flow directions file. By default, if the projection is well
        defined in the flow directions file. It is not necessary to provide the value of
        the EPSG. On the other hand, if the projection is not well defined in the flow directions file
        (i.e. in ASCII file). The **epsg** argument must be filled in.

    Returns
    -------
    dict
        A mesh dictionary that can be used to initialize the `Model` object.

    See Also
    --------
    Model: Primary data structure of the hydrological model `smash`.

    Examples
    --------
    >>> flwdir = smash.factory.load_dataset("flwdir")
    >>> mesh = smash.factory.generate_mesh(
    ... flwdir,
    ... x=[840_261, 826_553, 828_269],
    ... y=[6_457_807, 6_467_115, 6_469_198],
    ... area=[381.7 * 1e6, 107 * 1e6, 25.3 * 1e6],
    ... code=["V3524010", "V3515010", "V3517010"],
    ... )
    >>> mesh.keys()
    dict_keys(['dx', 'nrow', 'ncol', 'ng', 'nac', 'xmin', 'ymax',
               'flwdir', 'flwdst', 'flwacc', 'path', 'gauge_pos',
               'code', 'area', 'active_cell'])
    """

    if os.path.isfile(path):
        gdal.UseExceptions()
        ds_flwdir = gdal.Open(path)

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    if bbox is None:
        if x is None or y is None or area is None:
            raise ValueError(
                "'bbox' argument or 'x', 'y' and 'area' arguments must be defined"
            )

        else:
            return _get_mesh_from_xy(ds_flwdir, x, y, area, code, max_depth, epsg)

    else:
        return _get_mesh_from_bbox(ds_flwdir, bbox, epsg)
