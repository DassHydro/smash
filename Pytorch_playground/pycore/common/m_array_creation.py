import numpy as np

def arange_i(stt, stp, step):
    """
    Create an array of evenly spaced values within a specified interval.

    Parameters:
    - stt (float): Start of the interval.
    - stp (float): End of the interval.
    - step (float): Spacing between values.

    Returns:
    - res (ndarray): Array of evenly spaced values.
    """
    res = np.arange(stt, stp, step, dtype=float)
    return res

def arange_r(stt, stp, step):
    """
    Create an array of evenly spaced values within a specified interval.

    Parameters:
    - stt (float): Start of the interval.
    - stp (float): End of the interval.
    - step (float): Spacing between values.

    Returns:
    - res (ndarray): Array of evenly spaced values.
    """
    res = np.arange(stt, stp, step, dtype=float)
    return res

def linspace_i(stt, stp, n):
    """
    Create an array of evenly spaced values over a specified interval.

    Parameters:
    - stt (float): Start of the interval.
    - stp (float): End of the interval.
    - n (int): Number of values to generate.

    Returns:
    - res (ndarray): Array of evenly spaced values.
    """
    if n == 1:
        return np.array([stt], dtype=float)
    else:
        step = float(stp - stt) / float(n - 1)
        res = np.array([stt + i * step for i in range(n)], dtype=float)
        return res

def linspace_r(stt, stp, n):
    """
    Create an array of evenly spaced values over a specified interval.

    Parameters:
    - stt (float): Start of the interval.
    - stp (float): End of the interval.
    - n (int): Number of values to generate.

    Returns:
    - res (ndarray): Array of evenly spaced values.
    """
    if n == 1:
        return np.array([stt], dtype=float)
    else:
        step = (stp - stt) / float(n - 1)
        res = np.array([stt + i * step for i in range(n)], dtype=float)
        return res