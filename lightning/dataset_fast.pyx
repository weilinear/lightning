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

np.import_array()

cdef extern from "math.h":
   double exp(double)

cdef double powi(double base, int times):
    cdef double tmp = base, ret = 1.0

    cdef int t = times

    while t > 0:
        if t % 2 == 1:
            ret *= tmp
        tmp = tmp * tmp

        t /= 2

    return ret

cdef class Dataset:

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz):
        raise NotImplementedError()

    cpdef get_column(self, int j):
        cdef double* data
        cdef int* indices
        cdef int n_nz
        cdef np.npy_intp shape[1]

        self.get_column_ptr(j, &indices, &data, &n_nz)

        shape[0] = <np.npy_intp> self.n_samples
        indices_ = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, indices)
        data_ = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, data)

        return indices_, data_, n_nz

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

DEF LINEAR_KERNEL = 1
DEF POLY_KERNEL = 2
DEF RBF_KERNEL = 3

KERNELS = {"linear" : LINEAR_KERNEL,
           "poly" : POLY_KERNEL,
           "polynomial" : POLY_KERNEL,
           "rbf" : RBF_KERNEL}

cdef class KernelDataset(Dataset):

    def __init__(self,
                 np.ndarray[double, ndim=2, mode='c'] X,
                 np.ndarray[double, ndim=2, mode='c'] Y,
                 kernel="linear",
                 double gamma=0.1,
                 double coef0=1.0,
                 int degree=4):

        self.n_samples = X.shape[0]
        self.n_features = Y.shape[0]
        self.data = <double*> X.data

        self.n_features_Y = Y.shape[1]
        self.data_Y = <double*> Y.data

        self.kernel = KERNELS[kernel]
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

    def __cinit__(self,
                  np.ndarray[double, ndim=2, mode='c'] X,
                  np.ndarray[double, ndim=2, mode='c'] Y,
                  kernel="linear",
                  double gamma=0.1,
                  double coef0=1.0,
                  int degree=4):
        cdef int i
        cdef int n_samples = X.shape[0]
        self.indices = <int*> stdlib.malloc(sizeof(int) * n_samples)
        for i in xrange(n_samples):
            self.indices[i] = i
        self.cache = <double*> stdlib.malloc(sizeof(double) * n_samples)

    cdef void _linear_kernel(self,
                             int j,
                             double *out):
        cdef double dot = 0
        cdef int i, k
        cdef double* data_X
        cdef double* data_Y

        data_X = self.data
        data_Y = self.data_Y + j * self.n_features_Y

        for i in xrange(self.n_samples):
            dot = 0

            for k in xrange(self.n_features_Y):
                dot += data_X[k] * data_Y[k]

            out[i] = dot

            data_X += self.n_features_Y

    cdef void _poly_kernel(self,
                           int j,
                           double *out):
        cdef double dot = 0
        cdef int i, k
        cdef double* data_X
        cdef double* data_Y

        data_X = self.data
        data_Y = self.data_Y + j * self.n_features_Y

        for i in xrange(self.n_samples):
            dot = 0

            for k in xrange(self.n_features_Y):
                dot += data_X[k] * data_Y[k]

            out[i] = powi(self.coef0 + dot * self.gamma, self.degree)

            data_X += self.n_features_Y

    cdef void _rbf_kernel(self,
                          int j,
                          double *out):
        cdef double value
        cdef double diff
        cdef int i, k
        cdef double* data_X
        cdef double* data_Y

        data_X = self.data
        data_Y = self.data_Y + j * self.n_features_Y

        for i in xrange(self.n_samples):
            value = 0

            for k in xrange(self.n_features_Y):
                diff = data_X[k] - data_Y[k]
                value += diff * diff

            out[i] = exp(-self.gamma * value)

            data_X += self.n_features_Y

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz):
        if self.kernel == LINEAR_KERNEL:
            self._linear_kernel(j, self.cache)
        elif self.kernel == POLY_KERNEL:
            self._poly_kernel(j, self.cache)
        elif self.kernel == RBF_KERNEL:
            self._rbf_kernel(j, self.cache)
        else:
            raise ValueError("Unknown kernel!")

        indices[0] = self.indices
        data[0] = self.cache
        n_nz[0] = self.n_samples
