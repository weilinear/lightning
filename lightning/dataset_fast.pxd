# Author: Mathieu Blondel
# License: BSD

from libcpp.list cimport list
from libcpp.map cimport map

cdef class Dataset:

    cdef int n_samples
    cdef int n_features

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)

    cpdef get_column(self, int j)

    cpdef int get_n_samples(self)
    cpdef int get_n_features(self)


cdef class ContiguousDataset(Dataset):

    cdef int* indices
    cdef double* data
    cdef object X


cdef class FortranDataset(Dataset):

    cdef int* indices
    cdef double* data
    cdef object X

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)


cdef class CSCDataset(Dataset):

    cdef int* indices
    cdef double* data
    cdef int* indptr
    cdef object X

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)


cdef class KernelDataset(Dataset):

    # Input data
    cdef int* indices
    cdef double* data
    cdef int n_features_Y
    cdef double* data_Y

    # Kernel parameters
    cdef int kernel
    cdef double coef0
    cdef double gamma
    cdef int degree

    # Cache
    cdef double* cache
    cdef map[int, double*]* columns
    cdef int* n_computed
    cdef long capacity
    cdef int verbose
    cdef long size

    # Methods
    cdef void _linear_kernel(self, int j, double *out)
    cdef void _poly_kernel(self, int j, double *out)
    cdef void _rbf_kernel(self, int j, double *out)
    cdef void _kernel(self, int j, double *out)

    cdef void _create_column(self, int i)
    cdef void _clear_columns(self, int n)
    cdef double* _get_column(self, int j)
    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)
