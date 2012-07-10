# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

from libc cimport stdlib

import numpy as np
cimport numpy as np

cdef class Dataset:

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz):
        raise NotImplementedError()

    cpdef int get_n_samples(self):
        return self.n_samples

    cpdef int get_n_features(self):
        return self.n_features


cdef class FortranDataset(Dataset):

    def __init__(self, np.ndarray[double, ndim=2, mode='fortran'] X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.data = <double*> X.data

    def __cinit__(self, np.ndarray[double, ndim=2, mode='fortran'] X):
        cdef int i
        cdef int n_samples = X.shape[0]
        self.indices = <int*> stdlib.malloc(sizeof(int) * n_samples)
        for i in xrange(n_samples):
            self.indices[i] = i

    def __dealloc__(self):
        stdlib.free(self.indices)

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz):
        indices[0] = self.indices
        data[0] = self.data + j * self.n_samples
        n_nz[0] = self.n_samples


cdef class CSCDataset(Dataset):

    def __init__(self, X):
        cdef np.ndarray[double, ndim=1, mode='c'] X_data = X.data
        cdef np.ndarray[int, ndim=1, mode='c'] X_indices = X.indices
        cdef np.ndarray[int, ndim=1, mode='c'] X_indptr = X.indptr

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.data = <double*> X_data.data
        self.indices = <int*> X_indices.data
        self.indptr = <int*> X_indptr.data

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz):
        indices[0] = self.indices + self.indptr[j]
        data[0] = self.data + self.indptr[j]
        n_nz[0] = self.indptr[j + 1] - self.indptr[j]
