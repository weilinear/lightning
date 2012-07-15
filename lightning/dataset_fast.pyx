# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from cython.operator cimport postincrement as postinc
from cython.operator cimport predecrement as dec

from libcpp.list cimport list
from libcpp.map cimport map
from libc cimport stdlib

import numpy as np
cimport numpy as np
np.import_array()

from sklearn.utils.extmath import safe_sparse_dot

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

    def dot(self, coef):
        return NotImplementedError()


cdef class ContiguousDataset(Dataset):

    def __init__(self, np.ndarray[double, ndim=2, mode='c'] X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.data = <double*> X.data
        self.X = X

    def __cinit__(self, np.ndarray[double, ndim=2, mode='c'] X):
        cdef int i
        cdef int n_samples = X.shape[0]
        self.indices = <int*> stdlib.malloc(sizeof(int) * n_samples)
        for i in xrange(n_samples):
            self.indices[i] = i

    def __dealloc__(self):
        stdlib.free(self.indices)

    def dot(self, coef):
        return safe_sparse_dot(self.X, coef)


cdef class FortranDataset(Dataset):

    def __init__(self, np.ndarray[double, ndim=2, mode='fortran'] X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.data = <double*> X.data
        self.X = X

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

    def dot(self, coef):
        return safe_sparse_dot(self.X, coef)


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

        self.X = X

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz):
        indices[0] = self.indices + self.indptr[j]
        data[0] = self.data + self.indptr[j]
        n_nz[0] = self.indptr[j + 1] - self.indptr[j]

    def dot(self, coef):
        return safe_sparse_dot(self.X, coef)


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
                 int degree=4,
                 long capacity=500,
                 int mb=1,
                 int verbose=0):

        # Input data
        self.n_samples = X.shape[0]
        self.n_features = Y.shape[0]
        self.data = <double*> X.data
        self.n_features_Y = Y.shape[1]
        self.data_Y = <double*> Y.data

        # Kernel parameters
        self.kernel = KERNELS[kernel]
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

        # Cache
        if mb:
            self.capacity = capacity * (1 << 20)
        else:
            self.capacity = capacity
        self.verbose = verbose
        self.size = 0

    def __cinit__(self,
                  np.ndarray[double, ndim=2, mode='c'] X,
                  np.ndarray[double, ndim=2, mode='c'] Y,
                  kernel="linear",
                  double gamma=0.1,
                  double coef0=1.0,
                  int degree=4,
                  long capacity=500,
                  int mb=1,
                  int verbose=0):
        cdef int i
        cdef int n_samples = X.shape[0]

        # Allocate indices.
        self.indices = <int*> stdlib.malloc(sizeof(int) * n_samples)
        for i in xrange(n_samples):
            self.indices[i] = i

        # Allocate containers for cache.
        self.n_computed = <int*> stdlib.malloc(sizeof(int) * n_samples)
        self.columns = new map[int, double*]()

        for i in xrange(n_samples):
            self.n_computed[i] = 0

        self.cache = <double*> stdlib.malloc(sizeof(double) * n_samples)

    def __dealloc__(self):
        # De-allocate indices.
        stdlib.free(self.indices)

        # De-allocate containers for cache.
        self._clear_columns(self.n_samples)
        self.columns.clear()
        del self.columns
        stdlib.free(self.n_computed)

        stdlib.free(self.cache)

    cdef void _linear_kernel(self, int j, double *out):
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

    cdef void _poly_kernel(self, int j, double *out):
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

    cdef void _rbf_kernel(self, int j, double *out):
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

    cdef void _kernel(self, int j, double *out):
        if self.kernel == LINEAR_KERNEL:
            self._linear_kernel(j, out)
        elif self.kernel == POLY_KERNEL:
            self._poly_kernel(j, out)
        elif self.kernel == RBF_KERNEL:
            self._rbf_kernel(j, out)

    cdef void _create_column(self, int i):
        cdef int n_computed = self.n_computed[i]

        if n_computed != 0:
            return

        cdef int col_size = self.n_samples * sizeof(double)

        if self.size + col_size > self.capacity:
            if self.verbose >= 2:
                print "Empty cache by half"
            self._clear_columns(self.columns.size() / 2)

        self.columns[0][i] = <double*> stdlib.calloc(self.n_samples,
                                                     sizeof(double))
        self.size += col_size

    cdef void _clear_columns(self, int n):
        cdef map[int, double*].iterator it
        it = self.columns.begin()
        cdef int i = 0
        cdef int col_size

        while it != self.columns.end():
            col_size = self.n_samples * sizeof(double)
            stdlib.free(deref(it).second)
            self.n_computed[deref(it).first] = 0
            self.size -= col_size
            self.columns.erase(postinc(it))

            if i >= n - 1:
                break

            i += 1

    cdef double* _get_column(self, int j):
        cdef int i = 0

        if self.capacity == 0:
            self._kernel(j, self.cache)
            return self.cache

        cdef int n_computed = self.n_computed[j]

        self._create_column(j)

        cdef double* cache = &(self.columns[0][j][0])

        if n_computed == -1:
            # Full column is already computed.
            return cache
        #elif n_computed > 0:
            ## Some elements are already computed.
            #for i in xrange(self.n_samples):
                #if cache[i] == 0:
                    #out[i] = self.kernel.compute(X, i, Y, j)
                    #cache[i] = out[i]
                #else:
                    #out[i] = cache[i]
        else:
            # All elements must be computed.
            self._kernel(j, cache)

        self.n_computed[j] = -1

        return cache

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz):

        indices[0] = self.indices
        data[0] = self._get_column(j)
        n_nz[0] = self.n_samples

    def dot(self, coef):
        cdef int n_features = coef.shape[0]
        cdef int n_vectors = coef.shape[1]

        cdef np.ndarray[double, ndim=2, mode='c'] out
        out = np.zeros((self.n_samples, n_vectors), dtype=np.float64)

        cdef np.ndarray[double, ndim=2, mode='c'] coef_
        coef_ = np.ascontiguousarray(coef, dtype=np.float64)

        cdef int i, j, k

        for j in xrange(n_features):
            self._kernel(j, self.cache)

            for i in xrange(self.n_samples):
                for k in xrange(n_vectors):
                    out[i, k] += self.cache[i] * coef_[j, k]

        return out

