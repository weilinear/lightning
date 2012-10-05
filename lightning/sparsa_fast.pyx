# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import sys

import numpy as np
cimport numpy as np

from lightning.dataset_fast cimport Dataset

DEF LOWER = 1e-2
DEF UPPER = 1e9

cdef extern from "math.h":
   double fabs(double)
   double exp(double x)
   double log(double x)
   double sqrt(double x)

cdef extern from "float.h":
   double DBL_MAX

cdef class SquaredHinge:

    cpdef gradient(self,
                   np.ndarray[double, ndim=2, mode='c'] df,
                   Dataset X,
                   np.ndarray[int, ndim=1, mode='c'] y,
                   np.ndarray[double, ndim=2, mode='c'] G):

        cdef double* data
        cdef int* indices
        cdef int n_nz

        cdef int n_samples = df.shape[0]
        cdef int n_vectors = df.shape[1]
        cdef int i, k, j, jj
        cdef double update, tmp

        for i in xrange(n_samples):
            for k in xrange(n_vectors):
                if y[i] == k:
                    continue

                update = max(1 - df[i, y[i]] + df[i, k], 0)
                if update != 0:
                    X.get_row_ptr(i, &indices, &data, &n_nz)
                    for jj in xrange(n_nz):
                        j = indices[jj]
                        tmp = update * data[jj]
                        G[y[i], j] -= tmp
                        G[k, j] += tmp

        G *= 2

    cpdef objective(self,
                    np.ndarray[double, ndim=2, mode='c'] df,
                    np.ndarray[int, ndim=1, mode='c'] y):

        cdef int n_samples = df.shape[0]
        cdef int n_vectors = df.shape[1]

        cdef int i, k
        cdef double obj, value

        obj = 0

        for i in xrange(n_samples):
            for k in xrange(n_vectors):
                if y[i] == k:
                    continue
                value = max(1 - df[i, y[i]] + df[i, k], 0)
                obj += value * value

        return obj

