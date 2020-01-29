from libc.math cimport pow
from libcpp cimport bool
cimport cython
cimport numpy as np    
import numpy as np

# equivalence -> reported together
cdef double optimized_logical_equivalence (double a, double b):
    """Compute logical equivalence as double.
    This is a cdef function that can be called from within
    a Cython program, but not from Python.
    """
    nonzero_a = (a != 0)
    nonzero_b = (b != 0)
    return ((nonzero_a & nonzero_b) | (~nonzero_a & ~nonzero_b))

# implication
cdef double optimized_logical_implication (double a, double b):
    nonzero_a = (a != 0)
    nonzero_b = (b != 0)
    return ~(nonzero_a & ~nonzero_b)

cdef double optimized_logical_or (double a, double b):
    nonzero_a = (a != 0)
    nonzero_b = (b != 0)
    return (nonzero_a | nonzero_b)

cdef double optimized_logical_and (double a, double b):
    nonzero_a = (a != 0)
    nonzero_b = (b != 0)
    return (nonzero_a & nonzero_b)

cdef double summation (int[:] co, int n_rows):
    cdef int res = 0
    for i in range(0, n_rows):
        res += co[i]
    return res

# equivalence -> reported together
cpdef double logical_equivalence (double a, double b):
    return optimized_logical_equivalence(a, b)

# # implication
cpdef double logical_implication (double a, double b):
    return optimized_logical_implication(a, b)

cpdef double logical_or (double a, double b):
    return optimized_logical_or(a, b)

cpdef double logical_and (double a, double b):
    return optimized_logical_and(a, b)

cpdef double apply_sum (np.ndarray[np.npy_bool] co):
    assert (co.dtype == np.bool_)
    return summation(np.array(co, dtype=int), len(co))
