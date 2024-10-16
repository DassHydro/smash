import numpy as np

def reallocate1d_i(arr, nsz):
    """
    Reallocates a 1-dimensional integer array to a new size.

    Parameters:
    arr (ndarray): The input array to be reallocated.
    nsz (int): The new size of the array.

    Returns:
    ndarray: The reallocated array.

    """
    osz = len(arr)
    if osz == nsz:
        return arr
    elif osz < nsz:
        warr = np.zeros(nsz, dtype=int)
        warr[:osz] = arr
        return warr
    else:
        warr = np.zeros(nsz, dtype=int)
        warr[:nsz] = arr[:nsz]
        return warr

def reallocate1d_r(arr, nsz):
    """
    Reallocates a 1-dimensional float array to a new size.

    Parameters:
    arr (ndarray): The input array to be reallocated.
    nsz (int): The new size of the array.

    Returns:
    ndarray: The reallocated array.

    """
    osz = len(arr)
    if osz == nsz:
        return arr
    elif osz < nsz:
        warr = np.zeros(nsz, dtype=float)
        warr[:osz] = arr
        return warr
    else:
        warr = np.zeros(nsz, dtype=float)
        warr[:nsz] = arr[:nsz]
        return warr