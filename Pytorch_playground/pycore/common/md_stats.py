import numpy as np

def heap_sort(arr):
    """
    Sorts an array using the heap sort algorithm.

    Args:
        arr (list): The array to be sorted.

    Returns:
        None. The array is sorted in-place.
    """
    n = len(arr)
    l = n // 2 + 1
    ir = n
    while True:
        if l > 1:
            l -= 1
            arr_l = arr[l-1]
        else:
            arr_l = arr[ir-1]
            arr[ir-1] = arr[0]
            ir -= 1
            if ir == 1:
                arr[0] = arr_l
                return
        i = l
        j = l + l
        while j <= ir:
            if j < ir and arr[j-1] < arr[j]:
                j += 1
            if arr_l < arr[j-1]:
                arr[i-1] = arr[j-1]
                i = j
                j += j
            else:
                j = ir + 1
        arr[i-1] = arr_l

def quantile1d_r_scl(dat, p):
    """
    Calculates the quantile of a 1-dimensional array using the R-1 method.

    Args:
        dat (list): The input array.
        p (float): The quantile(s) to calculate. Can be a single value or a list of values.

    Returns:
        float or list: The calculated quantile(s).
    """
    n = len(dat)
    sorted_dat = dat.copy()
    heap_sort(sorted_dat)
    frac = (n - 1) * p + 1
    if frac <= 1:
        return sorted_dat[0]
    elif frac >= n:
        return sorted_dat[-1]
    else:
        q1 = sorted_dat[int(frac) - 1]
        q2 = sorted_dat[int(frac)]
        return q1 + (q2 - q1) * (frac - int(frac))

def quantile1d_r_1d(dat, p):
    """
    Calculates the quantiles of a 1-dimensional array using the R-1 method.

    Args:
        dat (list): The input array.
        p (list): The quantiles to calculate.

    Returns:
        list: The calculated quantiles.
    """
    n = len(dat)
    sorted_dat = dat.copy()
    heap_sort(sorted_dat)
    res = np.empty_like(p)
    for i in range(len(p)):
        frac = (n - 1) * p[i] + 1
        if frac <= 1:
            res[i] = sorted_dat[0]
        elif frac >= n:
            res[i] = sorted_dat[-1]
        else:
            q1 = sorted_dat[int(frac) - 1]
            q2 = sorted_dat[int(frac)]
            res[i] = q1 + (q2 - q1) * (frac - int(frac))
    return res