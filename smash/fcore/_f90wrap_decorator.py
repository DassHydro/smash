from __future__ import annotations

import functools

import numpy as np

import f90wrap
from smash.fcore import _libfcore


def f90wrap_getter_char(func):
    @functools.wraps(func)
    def wrapper(self):
        value = func(self)
        return value.decode().strip()

    return wrapper


#! setter is identical
def f90wrap_setter_char(func):
    @functools.wraps(func)
    def wrapper(self, value):
        func(self, value)

    return wrapper


def f90wrap_getter_char_array(func):
    @functools.wraps(func)
    def wrapper(self):
        value = func(self)
        shape = value.shape
        arr = np.empty(shape=shape[1:], dtype=f"U{shape[0]}")
        for idx in np.ndindex(shape[1:]):
            slc = (slice(shape[0]), *idx)
            arr[idx] = value[slc].tobytes(order="F").decode().strip()
        return arr

    return wrapper


#! setter completly rewrite (func is not called)
def f90wrap_setter_char_array(func):
    @functools.wraps(func)
    def wrapper(self, value):
        func_name = func.__name__
        class_name = self.__class__.__name__.lower()

        getter = getattr(_libfcore, f"f90wrap_{class_name}__array__{func_name}")

        array_ndim, _, array_shape, array_handle = getter(self._handle)

        shape = tuple(s for i, s in enumerate(array_shape) if i < array_ndim)

        if isinstance(value, (str, list, tuple)):
            value = np.array(value)

        elif isinstance(value, np.ndarray):
            value = value.astype("U")

        else:
            raise TypeError(f"{func_name} attribute must be str or ListLike object, not {type(value)}")

        if value.shape != shape[1:] and value.size != 1:
            raise ValueError(f"could not broadcast input array from shape {value.shape} into {shape[1:]}")

        else:
            arr = np.empty(shape, dtype=np.uint8)
            arr.fill(np.uint8(32))

            if value.size == 1:
                item = list(value.item().encode("ascii"))
                litem = len(item)
                # length of char greater than char limit in Fortran
                if litem > shape[0]:
                    raise ValueError(
                        f"the length of '{value.item()}' ({litem}) exceeds the maximum limit ({shape[0]})"
                    )
                for idx in np.ndindex(shape[1:]):
                    slc = (slice(litem), *idx)
                    arr[slc] = item
            else:
                for idx in np.ndindex(shape[1:]):
                    item = list(value[idx].encode("ascii"))
                    litem = len(item)
                    # length of char greater than char limit in Fortran
                    if litem > shape[0]:
                        raise ValueError(
                            f"the length of '{value[idx]}' ({litem}) exceeds the maximum limit ({shape[0]})"
                        )
                    slc = (slice(litem), *idx)
                    arr[slc] = item

        if array_handle in self._arrays:
            ptr = self._arrays[array_handle]
        else:
            ptr = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t, self._handle, getter)
            self._arrays[array_handle] = ptr

        ptr[...] = arr

    return wrapper


def f90wrap_getter_index(func):
    @functools.wraps(func)
    def wrapper(self):
        value = func(self)
        return np.add(value, -1)

    return wrapper


def f90wrap_setter_index(func):
    @functools.wraps(func)
    def wrapper(self, value):
        value = np.add(value, 1)
        func(self, value)

    return wrapper


def f90wrap_getter_index_array(func):
    return f90wrap_getter_index(func)


#! setter completly rewrite (func is not called)
def f90wrap_setter_index_array(func):
    @functools.wraps(func)
    def wrapper(self, value):
        func_name = func.__name__
        class_name = self.__class__.__name__.lower()

        getter = getattr(_libfcore, f"f90wrap_{class_name}__array__{func_name}")

        _, _, _, array_handle = getter(self._handle)

        arr = np.add(value, 1)

        if array_handle in self._arrays:
            ptr = self._arrays[array_handle]
        else:
            ptr = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t, self._handle, getter)
            self._arrays[array_handle] = ptr

        ptr[...] = arr

    return wrapper
