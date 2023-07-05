from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import Timestamp


def _map_dict_to_object(dct: dict, obj: object):
    for key, value in dct.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def _index_containing_substring(the_list: list, substring: str):
    for i, s in enumerate(the_list):
        if substring in s:
            return i
    return -1


def _adjust_left_files_by_date(files: list[str], date_range: Timestamp):
    n = 0
    ind = -1
    while ind == -1:
        ind = _index_containing_substring(files, date_range[n].strftime("%Y%m%d%H%M"))

        n += 1

    return files[ind:]


def _check_unknown_options(type_check: str, unknown_options: dict):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))

        raise KeyError(f"Unknown {type_check} options: '{msg}'")
