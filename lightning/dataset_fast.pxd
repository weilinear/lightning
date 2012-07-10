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
