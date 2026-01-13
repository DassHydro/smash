import numpy as np
import pytest

from smash.fcore._mwd_mesh import MeshDT
from smash.fcore._mwd_setup import SetupDT

# TODO (FC): These are the minimum tests that can be performed without creating specific test cases with
# Fortran files to be compiled for the tests.


def test_char():
    setup = SetupDT(0)
    snow_module = "Picsou"
    setup.snow_module = snow_module
    assert setup.snow_module == snow_module, "char.value"


def test_char_array():
    nd = 2
    setup = SetupDT(nd)

    descriptor_name = "Donald"
    setup.descriptor_name = descriptor_name
    assert np.array_equal(setup.descriptor_name, np.array(nd * [descriptor_name])), "char_array.str_value"

    descriptor_name = ["Donald"]
    setup.descriptor_name = descriptor_name
    assert np.array_equal(setup.descriptor_name, np.array(nd * descriptor_name)), "char_array.list_1_value"

    descriptor_name = nd * ["Donald"]
    setup.descriptor_name = descriptor_name
    assert np.array_equal(setup.descriptor_name, np.array(descriptor_name)), "char_array.list_nd_value"

    descriptor_name = np.array(nd * ["Donald"])
    setup.descriptor_name = descriptor_name
    assert np.array_equal(setup.descriptor_name, descriptor_name), "char_array.ndarray_nd_value"

    with pytest.raises(ValueError, match="could not broadcast"):
        descriptor_name = np.array(42 * ["Donald"])
        setup.descriptor_name = descriptor_name

    with pytest.raises(TypeError, match="attribute must be str or ListLike object"):
        descriptor_name = 42
        setup.descriptor_name = descriptor_name

    with pytest.raises(ValueError, match="exceeds the maximum limit"):
        descriptor_name = (
            "Miiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"
            "iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiickey"
        )
        setup.descriptor_name = descriptor_name


def test_index_array():
    setup = SetupDT(0)
    mesh = MeshDT(setup, 10, 10, 2, 1)

    # % Initialized to -99 in Fortan. Should get Fortran value minus 1 in Python
    assert np.all(mesh.gauge_pos == -100), "index_array.fortran_value"

    mesh.gauge_pos = 0
    assert np.all(mesh.gauge_pos == 0), "index_array.python_value"
