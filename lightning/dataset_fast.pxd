# Author: Mathieu Blondel
# License: BSD

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


cdef class FortranDataset(Dataset):

    cdef int* indices
    cdef double* data

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)


cdef class CSCDataset(Dataset):

    cdef int* indices
    cdef double* data
    cdef int* indptr

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)


cdef class KernelDataset(Dataset):

    cdef int* indices
    cdef double* data

    cdef int n_features_Y
    cdef double* data_Y

    cdef double* cache

    cdef int kernel
    cdef double coef0
    cdef double gamma
    cdef int degree

    cdef void _linear_kernel(self,
                             int j,
                             double *out)
    cdef void _poly_kernel(self,
                           int j,
                           double *out)
    cdef void _rbf_kernel(self,
                          int j,
                          double *out)

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)
