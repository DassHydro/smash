import functools
import numpy as np
import f90wrap

from smash.solver import _solver


def char_getter_handler(func):
    @functools.wraps(func)
    def wrapper(self):
        value = func(self)
        return value.decode().strip()

    return wrapper


def char_array_getter_handler(func):
    @functools.wraps(func)
    def wrapper(self):
        value = func(self)
        return np.array(value.tobytes(order="F").decode().split())

    return wrapper


#! setter completly rewrite (func is not called)
def char_array_setter_handler(func):
    @functools.wraps(func)
    def wrapper(self, value):

        func_name = func.__name__
        class_name = self.__class__.__name__.lower()

        getter = eval(f"_solver.f90wrap_{class_name}__array__{func_name}")

        array_ndim, arra_type, array_shape, array_handle = getter(self._handle)

        shape = tuple(s for i, s in enumerate(array_shape) if i < array_ndim)

        if isinstance(value, (list, tuple)):

            value = np.array(value)

        elif isinstance(value, np.ndarray):

            value = value.astype("U")

        else:

            raise TypeError(
                f"{func_name} attribute must be list-like object, not {type(value)}"
            )

        if value.size != shape[1]:

            raise ValueError(
                f"could not broadcast input array from shape {value.shape} into {(shape[1],)}"
            )

        else:

            arr = np.zeros(shape, dtype="uint8") + np.uint8(32)

            for i, el in enumerate(value):

                arr[0 : len(el), i] = [ord(l) for l in el]

        if array_handle in self._arrays:
            ptr = self._arrays[array_handle]
        else:
            ptr = f90wrap.runtime.get_array(
                f90wrap.runtime.sizeof_fortran_t, self._handle, getter
            )
            self._arrays[array_handle] = ptr

        ptr[...] = arr

    return wrapper


def getter_index_handler(func):
    @functools.wraps(func)
    def wrapper(self):
        value = func(self)
        return np.add(value, -1)
        
    return wrapper

#! setter completly rewrite (func is not called)
def setter_index_handler(func):
    @functools.wraps(func)
    def wrapper(self, value):
        
        func_name = func.__name__
        class_name = self.__class__.__name__.lower()

        getter = eval(f"_solver.f90wrap_{class_name}__array__{func_name}")
        
        array_ndim, arra_type, array_shape, array_handle = getter(self._handle)
        
        arr = np.add(value, 1)

        if array_handle in self._arrays:
            ptr = self._arrays[array_handle]
        else:
            ptr = f90wrap.runtime.get_array(
                f90wrap.runtime.sizeof_fortran_t, self._handle, getter
            )
            self._arrays[array_handle] = ptr
            
        ptr[...] = arr
        
    return wrapper

