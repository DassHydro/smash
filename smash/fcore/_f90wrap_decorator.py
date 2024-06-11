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
        return np.array(value.tobytes(order="F").decode().split())

    return wrapper


#! setter completly rewrite (func is not called)
def f90wrap_setter_char_array(func):
    @functools.wraps(func)
    def wrapper(self, value):
        func_name = func.__name__
        class_name = self.__class__.__name__.lower()

        getter = getattr(_libfcore, f"f90wrap_{class_name}__array__{func_name}")

        array_ndim, array_type, array_shape, array_handle = getter(self._handle)

        shape = tuple(s for i, s in enumerate(array_shape) if i < array_ndim)

        if isinstance(value, (list, tuple)):
            value = np.array(value)

        elif isinstance(value, np.ndarray):
            value = value.astype("U")

        else:
            raise TypeError(f"{func_name} attribute must be list-like object, not {type(value)}")

        if value.size != shape[1]:
            raise ValueError(f"could not broadcast input array from shape {value.shape} into {(shape[1],)}")

        else:
            arr = np.zeros(shape, dtype="uint8") + np.uint8(32)

            for i, el in enumerate(value):
                arr[0 : len(el), i] = [ord(char) for char in el]

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

        array_ndim, array_type, array_shape, array_handle = getter(self._handle)

        arr = np.add(value, 1)

        if array_handle in self._arrays:
            ptr = self._arrays[array_handle]
        else:
            ptr = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t, self._handle, getter)
            self._arrays[array_handle] = ptr

        ptr[...] = arr

    return wrapper
