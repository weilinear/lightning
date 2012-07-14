# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import sys

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from cython.operator cimport predecrement as dec

from libcpp.list cimport list
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from lightning.kernel_fast cimport KernelCache
from lightning.kernel_fast cimport Kernel
from lightning.select_fast cimport get_select_method
from lightning.select_fast cimport select_sv_precomputed
from lightning.random.random_fast cimport RandomState
from lightning.dataset_fast cimport Dataset

cdef extern from "math.h":
   double fabs(double)
   double exp(double x)
   double log(double x)
   double sqrt(double x)

cdef extern from "float.h":
   double DBL_MAX

cdef class LossFunction:

    cdef int max_steps

    # L2 regularization

    cdef void solve_l2(self,
                       int j,
                       double C,
                       double *w,
                       int *indices,
                       double *data,
                       int n_nz,
                       double *col,
                       double *y,
                       double *b,
                       double *Dp):

        cdef double sigma = 0.01
        cdef double beta = 0.5
        cdef double bound, Dpp, Dj_zero, z, d
        cdef int i, ii, step
        cdef double z_diff, z_old, Dj_z, cond

        # Compute derivatives
        self.derivatives(j, C, indices, data, n_nz, col, y, b,
                         Dp, &Dpp, &Dj_zero, &bound)

        Dp[0] = w[j] + Dp[0]
        Dpp = 1 + Dpp

        if bound != 0:
            bound = (2 * C * bound + 1) / 2.0 + sigma

        if fabs(Dp[0]/Dpp) <= 1e-12:
            return

        d = -Dp[0] / Dpp

        # Perform line search
        z_old = 0
        z = d

        step = 1
        while True:
            z_diff = z_old - z

            # lambda <= Dpp/bound is equivalent to Dp/z <= -bound
            if bound > 0 and Dp[0]/z + bound <= 0:
                for ii in xrange(n_nz):
                    i = indices[ii]
                    b[i] += z_diff * col[i]
                break

            # Update old predictions
            self.update(j, z_diff, C, indices, data, n_nz,
                        col, y, b, &Dj_z)

            if step == self.max_steps:
                break

            #   0.5 * (w + z e_j)^T (w + z e_j)
            # = 0.5 * w^T w + w_j z + 0.5 z^2
            cond = w[j] * z + (0.5 + sigma) * z * z
            cond += Dj_z - Dj_zero
            if cond <= 0:
                break

            z_old = z
            z *= beta
            step += 1

        w[j] += z

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *col,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L,
                          double *bound):
        raise NotImplementedError()

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *col,
                     double *y,
                     double *b,
                     double *L_new):
        raise NotImplementedError()

    # L1 regularization

    cdef int solve_l1(self,
                      int j,
                      double C,
                      double *w,
                      int n_samples,
                      int *indices,
                      double *data,
                      int n_nz,
                      double *col,
                      double *y,
                      double *b,
                      double Lpmax_old,
                      double *violation,
                      int *n_sv):
        cdef double Lj_zero = 0
        cdef double Lp = 0
        cdef double Lpp = 0
        cdef double xj_sq = 0
        cdef double Lpp_wj, d, wj_abs
        cdef double cond
        cdef double appxcond = 0
        cdef double Lj_z
        cdef int step

        cdef double sigma = 0.01
        cdef double beta = 0.5

        # Compute derivatives
        self.derivatives(j, C, indices, data, n_nz, col, y, b,
                         &Lp, &Lpp, &Lj_zero, &xj_sq)

        xj_sq *= C
        Lpp = max(Lpp, 1e-12)

        Lp_p = Lp + 1
        Lp_n = Lp - 1
        violation[0] = 0

        # Shrinking.
        if w[j] == 0:
            if Lp_p < 0:
                violation[0] = -Lp_p
            elif Lp_n > 0:
                violation[0] = Lp_n
            elif Lp_p > Lpmax_old / n_samples and Lp_n < -Lpmax_old / n_samples:
                # Shrink!
                return 1
        elif w[j] > 0:
            violation[0] = fabs(Lp_p)
        else:
            violation[0] = fabs(Lp_n)

        # Obtain Newton direction d.
        Lpp_wj = Lpp * w[j]
        if Lp_p <= Lpp_wj:
            d = -Lp_p / Lpp
        elif Lp_n >= Lpp_wj:
            d = -Lp_n / Lpp
        else:
            d = -w[j]

        if fabs(d) < 1.0e-12:
            return 0

        wj_abs = fabs(w[j])
        delta = fabs(w[j] + d) - wj_abs + Lp * d
        z_old = 0
        z = d

        # Check z = lambda*d for lambda = 1, beta, beta^2 such that
        # sufficient decrease condition is met.
        step = 1
        while True:
            # Reversed because of the minus in b[i] = 1 - y_i w^T x_i.
            z_diff = z_old - z
            cond = fabs(w[j] + z) - wj_abs - sigma * delta

            appxcond = xj_sq * z * z + Lp * z + cond

            # Avoid line search if possible.
            if xj_sq > 0 and appxcond <= 0:
                for ii in xrange(n_nz):
                    i = indices[ii]
                    # Need to remove the old z and had the new one.
                    b[i] += z_diff * col[i]
                break

            # Compute objective function value.
            self.update(j, z_diff, C, indices, data, n_nz, col, y, b, &Lj_z)

            if step == self.max_steps:
                break

            # Check stopping condition.
            cond = cond + Lj_z - Lj_zero
            if cond <= 0:
                break

            z_old = z
            z *= beta
            delta *= beta
            step += 1

        # end for num_linesearch

        if w[j] == 0 and z != 0:
            n_sv[0] += 1
        elif z != 0 and w[j] == -z:
            n_sv[0] -= 1

        w[j] += z

        return 0

    # L1/L2 regularization

    cdef void solve_l1l2(self,
                         int j,
                         double C,
                         np.ndarray[double, ndim=2, mode='c'] w,
                         int n_vectors,
                         int* indices,
                         double *data,
                         int n_nz,
                         int* y,
                         np.ndarray[double, ndim=2, mode='fortran'] Y,
                         int multiclass,
                         np.ndarray[double, ndim=2, mode='c'] b,
                         double *g,
                         double *d,
                         double *d_old,
                         double* Z):

        cdef int i, k, ii, n
        cdef double scaling, delta, L, R_j, Lpp_max
        cdef double tmp, L_new, R_j_new
        cdef double L_tmp, bound, Lpp_tmp

        cdef double beta = 0.5
        cdef double sigma = 0.01
        cdef int n_samples = Y.shape[0]
        cdef double* y_ptr
        cdef double* b_ptr
        cdef double* Z_ptr
        cdef double z_diff

        if multiclass:
            self.derivatives_mc(j, C, n_vectors, indices, data, n_nz,
                                y, b, g, Z, &L, &Lpp_max)
        else:
            L = 0
            Lpp_max = -DBL_MAX
            y_ptr = <double*>Y.data
            b_ptr = <double*>b.data
            Z_ptr = Z

            for k in xrange(n_vectors):
                self.derivatives(j, C, indices, data, n_nz, Z_ptr, y_ptr,
                                 b_ptr, &g[k], &Lpp_tmp, &L_tmp, &bound)
                L += L_tmp
                Lpp_max = max(Lpp_max, Lpp_tmp)
                y_ptr += n_samples
                b_ptr += n_samples
                Z_ptr += n_samples

            Lpp_max = min(max(Lpp_max, 1e-9), 1e9)

        # Compute vector to be projected.
        for k in xrange(n_vectors):
            d[k] = w[k, j] - g[k] / Lpp_max

        # Project.
        scaling = 0
        R_j = 0
        for k in xrange(n_vectors):
            R_j += w[k, j] * w[k, j]
            scaling += d[k] * d[k]

        scaling = 1 - 1 / (Lpp_max * sqrt(scaling))

        if scaling < 0:
            scaling = 0

        R_j = sqrt(R_j)

        delta = 0
        for k in xrange(n_vectors):
            d_old[k] = 0
            # Difference between new and old solution.
            d[k] = scaling * d[k] - w[k, j]
            delta += d[k] * g[k]

        # Perform line search.
        for n in xrange(self.max_steps):

            # Update predictions, normalizations and objective value.
            if multiclass:
                self.update_mc(C, n_vectors, indices, data, n_nz,
                               y, b, d, d_old, Z, &L_new)
            else:
                L_new = 0
                y_ptr = <double*>Y.data
                b_ptr = <double*>b.data
                Z_ptr = Z

                for k in xrange(n_vectors):
                    z_diff = d_old[k] - d[k]
                    self.update(j, z_diff, C, indices, data, n_nz,
                                Z_ptr, y_ptr, b_ptr, &L_tmp)
                    L_new += L_tmp
                    y_ptr += n_samples
                    b_ptr += n_samples
                    Z_ptr += n_samples

            # Compute regularization term.
            R_j_new = 0
            for k in xrange(n_vectors):
                tmp = w[k, j] + d[k]
                R_j_new += tmp * tmp
            R_j_new = sqrt(R_j_new)
            # R_new = R - R_j + R_j_new

            if n == 0:
                delta += R_j_new - R_j
                delta *= sigma

            # Check decrease condition
            if L_new - L + R_j_new - R_j <= delta:
                break
            else:
                delta *= beta
                for k in xrange(n_vectors):
                    d_old[k] = d[k]
                    d[k] *= beta

        # Update solution
        for k in xrange(n_vectors):
            w[k, j] += d[k]

    cdef void derivatives_mc(self,
                             int j,
                             double C,
                             int n_vectors,
                             int* indices,
                             double *data,
                             int n_nz,
                             int* y,
                             np.ndarray[double, ndim=2, mode='c'] b,
                             double* g,
                             double* Z,
                             double* L,
                             double* Lpp_max):
        raise NotImplementedError()

    cdef void update_mc(self,
                        double C,
                        int n_vectors,
                        int* indices,
                        double *data,
                        int n_nz,
                        int* y,
                        np.ndarray[double, ndim=2, mode='c'] b,
                        double *d,
                        double *d_old,
                        double* Z,
                        double* L_new):
        raise NotImplementedError()


cdef class Squared(LossFunction):

    def __init__(self):
        self.max_steps = 1

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *col,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L,
                          double *bound):
        cdef int ii, i

        Lp[0] = 0
        Lpp[0] = 0
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            Lpp[0] += data[ii] * data[ii]
            Lp[0] += b[i] * data[ii]
            L[0] += b[i] * b[i]

        Lpp[0] = 2 * C * Lpp[0]
        Lp[0] = 2 * C * Lp[0]
        L[0] *= C
        bound[0] = 0

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *col,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int ii, i

        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b[i] -= z_diff * data[ii]
            L_new[0] += b[i] * b[i]

        L_new[0] *= C


cdef class SquaredHinge(LossFunction):

    def __init__(self, int max_steps=30):
        self.max_steps = max_steps

    # Binary

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *col,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L,
                          double *bound):
        cdef int i, ii
        cdef double xj_sq = 0
        cdef double val

        Lp[0] = 0
        Lpp[0] = 0
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            val = data[ii] * y[i]
            col[i] = val
            xj_sq += val * val

            if b[i] > 0:
                Lp[0] -= b[i] * val
                Lpp[0] += val * val
                L[0] += b[i] * b[i]

        Lp[0] = 2 * C * Lp[0]
        Lpp[0] = 2 * C * Lpp[0]
        bound[0] = xj_sq
        L[0] *= C

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *col,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int i, ii
        cdef double b_new

        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b_new = b[i] + z_diff * col[i]
            b[i] = b_new
            if b_new > 0:
                L_new[0] += b_new * b_new

        L_new[0] *= C

    # Multiclass

    cdef void derivatives_mc(self,
                             int j,
                             double C,
                             int n_vectors,
                             int* indices,
                             double *data,
                             int n_nz,
                             int* y,
                             np.ndarray[double, ndim=2, mode='c'] b,
                             double* g,
                             double* Z,
                             double* L,
                             double* Lpp_max):

        cdef int ii, i, k
        cdef double tmp

        # Compute objective value, gradient and largest second derivative.
        Lpp_max[0] = 0
        L[0] = 0

        for k in xrange(n_vectors):
            g[k] = 0

            for ii in xrange(n_nz):
                i = indices[ii]

                if y[i] == k:
                    continue

                if b[k, i] > 0:
                    L[0] += b[k, i] * b[k, i]
                    tmp = b[k, i] * data[ii]
                    g[y[i]] -= tmp
                    g[k] += tmp
                    Lpp_max[0] += data[ii] * data[ii]

        for k in xrange(n_vectors):
            g[k] *= 2 * C

        L[0] *= C
        Lpp_max[0] *= 2 * C
        Lpp_max[0] = min(max(Lpp_max[0], 1e-9), 1e9)

    cdef void update_mc(self,
                        double C,
                        int n_vectors,
                        int* indices,
                        double *data,
                        int n_nz,
                        int* y,
                        np.ndarray[double, ndim=2, mode='c'] b,
                        double *d,
                        double *d_old,
                        double* Z,
                        double* L_new):

        cdef int ii, i, k
        cdef double tmp, b_new

        L_new[0] = 0
        for ii in xrange(n_nz):
            i = indices[ii]

            tmp = d_old[y[i]] - d[y[i]]

            for k in xrange(n_vectors):
                if k == y[i]:
                    continue

                b_new = b[k, i] + (tmp - (d_old[k] - d[k])) * data[ii]
                b[k, i] = b_new
                if b_new > 0:
                    L_new[0] += b_new * b_new

        L_new[0] *= C


cdef class ModifiedHuber(LossFunction):

    def __init__(self, int max_steps=30):
        self.max_steps = max_steps

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *col,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L,
                          double *bound):
        cdef int i, ii
        cdef double xj_sq = 0
        cdef double val

        Lp[0] = 0
        Lpp[0] = 0
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            val = data[ii] * y[i]
            col[i] = val
            xj_sq += val * val

            if b[i] > 2:
                Lp[0] -= 2 * val
                # -4 yp = 4 (b[i] - 1)
                L[0] += 4 * (b[i] - 1)
            elif b[i] > 0:
                Lp[0] -= b[i] * val
                Lpp[0] += val * val
                L[0] += b[i] * b[i]

        Lp[0] = 2 * C * Lp[0]
        Lpp[0] = 2 * C * Lpp[0]
        bound[0] = 0
        L[0] *= C

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *col,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int i, ii
        cdef double b_new

        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b_new = b[i] + z_diff * col[i]
            b[i] = b_new

            if b_new > 2:
                L_new[0] += 4 * (b[i] - 1)
            elif b_new > 0:
                L_new[0] += b_new * b_new

        L_new[0] *= C


cdef class Log(LossFunction):

    def __init__(self, int max_steps=30):
        self.max_steps = max_steps

    # Binary

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *col,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L,
                          double *bound):
        cdef int i, ii
        cdef double xj_sq = 0
        cdef double val, tau, exppred

        Lp[0] = 0
        Lpp[0] = 0
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            val = data[ii] * y[i]
            col[i] = val

            exppred = 1 + 1 / b[i]
            tau = 1 / exppred
            Lp[0] += val * (tau - 1)
            Lpp[0] += val * val * tau * (1 - tau)
            L[0] += log(exppred)

        Lp[0] = C * Lp[0]
        Lpp[0] = C * Lpp[0]
        L[0] *= C
        bound[0] = 0


    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *col,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int i, ii
        cdef double exppred

        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b[i] /= exp(z_diff * col[i])
            exppred = 1 + 1 / b[i]
            L_new[0] += log(exppred)

        L_new[0] *= C


    # Multiclass

    cdef void derivatives_mc(self,
                             int j,
                             double C,
                             int n_vectors,
                             int* indices,
                             double *data,
                             int n_nz,
                             int* y,
                             np.ndarray[double, ndim=2, mode='c'] b,
                             double *g,
                             double* Z,
                             double* L,
                             double* Lpp_max):

        cdef int ii, i, k
        cdef double Lpp, tmp

        # Compute normalization and objective value.
        L[0] = 0
        for ii in xrange(n_nz):
            i = indices[ii]
            Z[i] = 0
            for k in xrange(n_vectors):
                Z[i] += b[k, i]
            L[0] += log(Z[i])
        L[0] *= C

        # Compute gradient and largest second derivative.
        Lpp_max[0] = -DBL_MAX

        for k in xrange(n_vectors):
            g[k] = 0
            Lpp = 0

            for ii in xrange(n_nz):
                i = indices[ii]

                if y[i] == k:
                    continue

                if Z[i] > 0:
                    tmp = b[k, i] / Z[i]
                    g[k] += tmp * data[ii]
                    Lpp += data[ii] * data[ii] * tmp * (1 - tmp)

            g[k] *= C
            Lpp *= C
            Lpp_max[0] = max(Lpp, Lpp_max[0])

        Lpp_max[0] = min(max(Lpp_max[0], 1e-9), 1e9)

    cdef void update_mc(self,
                        double C,
                        int n_vectors,
                        int* indices,
                        double *data,
                        int n_nz,
                        int* y,
                        np.ndarray[double, ndim=2, mode='c'] b,
                        double* d,
                        double* d_old,
                        double* Z,
                        double* L_new):
        cdef int i, ii, k
        cdef double tmp

        L_new[0] = 0
        for ii in xrange(n_nz):
            i = indices[ii]
            tmp = d_old[y[i]] - d[y[i]]
            Z[i] = 0

            for k in xrange(n_vectors):
                if y[i] != k:
                    b[k, i] *= exp((d[k] - d_old[k] + tmp) * data[ii])
                Z[i] += b[k, i]

            L_new[0] += log(Z[i])

        L_new[0] *= C


def _primal_cd_l1r(self,
                   np.ndarray[double, ndim=1, mode='c'] w,
                   np.ndarray[double, ndim=1, mode='c'] b,
                   Dataset X,
                   np.ndarray[double, ndim=1] y,
                   np.ndarray[int, ndim=1, mode='c'] index,
                   LossFunction loss,
                   selection,
                   int search_size,
                   termination,
                   int n_components,
                   double C,
                   int max_iter,
                   RandomState rs,
                   double tol,
                   callback,
                   int verbose):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = index.shape[0]

    cdef int j, s, t
    cdef int active_size = n_features

    cdef double Lpmax_old = DBL_MAX
    cdef double Lpmax_new
    cdef double Lpmax_init
    cdef double violation

    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)

    cdef int check_n_sv = termination == "n_components"
    cdef int check_convergence = termination == "convergence"
    cdef int stop = 0
    cdef int select_method = get_select_method(selection)
    cdef int permute = selection == "permute"
    cdef int has_callback = callback is not None

    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef int n_sv = 0
    cdef int shrink = 0

    # FIXME: would be better to store the support indices in the class.
    for j in xrange(n_features):
        if w[j] != 0:
            n_sv += 1

    for t in xrange(max_iter):
        if verbose >= 1:
            print "\nIteration", t

        Lpmax_new = 0

        if permute:
            rs.shuffle(index[:active_size])

        s = 0

        while s < active_size:
            if permute:
                j = index[s]
            else:
                j = select_sv_precomputed(index, search_size,
                                          active_size, select_method, b, rs)

            X.get_column_ptr(j, &indices, &data, &n_nz)

            loss.solve_l1(j, C, <double*>w.data, n_samples,
                          indices, data, n_nz, <double*>col.data,
                          <double*>y.data, <double*>b.data,
                          Lpmax_old, &violation, &n_sv)

            Lpmax_new = max(Lpmax_new, violation)

            if shrink:
                active_size -= 1
                index[s], index[active_size] = index[active_size], index[s]
                continue

            # Exit if necessary.
            if check_n_sv and n_sv >= n_components:
                stop = 1
                break

            # Callback
            if has_callback and s % 100 == 0:
                ret = callback(self)
                if ret is not None:
                    stop = 1
                    break

            if verbose >= 1 and s % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

            s += 1
        # while active_size

        if stop:
            break

        if t == 0:
            Lpmax_init = Lpmax_new

        if check_convergence and Lpmax_new <= tol * Lpmax_init:
            if active_size == n_features:
                if verbose >= 1:
                    print "\nConverged at iteration", t
                break
            else:
                active_size = n_features
                Lpmax_old = DBL_MAX
                continue

        Lpmax_old = Lpmax_new

    # end for while max_iter

    if verbose >= 1:
        print

    return w


def _primal_cd_l1l2r(self,
                     np.ndarray[double, ndim=2, mode='c'] w,
                     np.ndarray[double, ndim=2, mode='c'] b,
                     Dataset X,
                     np.ndarray[int, ndim=1] y,
                     np.ndarray[double, ndim=2, mode='fortran'] Y,
                     int multiclass,
                     np.ndarray[int, ndim=1, mode='c'] index,
                     LossFunction loss,
                     double C,
                     int max_iter,
                     RandomState rs,
                     double tol,
                     callback,
                     int verbose):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = index.shape[0]
    cdef Py_ssize_t n_vectors = w.shape[0]

    cdef int t, s, i, j, k, n
    cdef int active_size = n_features

    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef np.ndarray[double, ndim=1, mode='c'] g
    g = np.zeros(n_vectors, dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] d
    d = np.zeros(n_vectors, dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] d_old
    d_old = np.zeros(n_vectors, dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode='c'] Z
    if multiclass:
        Z = np.zeros((n_samples, 1), dtype=np.float64)
    else:
        Z = np.zeros((n_vectors, n_samples), dtype=np.float64)

    for t in xrange(max_iter):
        if verbose >= 1:
            print "\nIteration", t

        rs.shuffle(index[:active_size])

        for s in xrange(active_size):
            j = index[s]

            X.get_column_ptr(j, &indices, &data, &n_nz)

            loss.solve_l1l2(j, C, w, n_vectors,
                            indices, data, n_nz,
                            <int*>y.data, Y, multiclass,
                            b, <double*>g.data, <double*>d.data,
                            <double*>d_old.data, <double*>Z.data)


def _primal_cd_l2r(self,
                   np.ndarray[double, ndim=1, mode='c'] w,
                   np.ndarray[double, ndim=1, mode='c'] b,
                   Dataset X,
                   np.ndarray[double, ndim=1] y,
                   np.ndarray[int, ndim=1, mode='c'] index,
                   LossFunction loss,
                   selection,
                   int search_size,
                   termination,
                   int n_components,
                   double C,
                   int max_iter,
                   RandomState rs,
                   double tol,
                   callback,
                   int verbose):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = index.shape[0]

    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef int i, j, s, t
    cdef double Dp, Dpmax

    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)

    cdef int check_n_sv = termination == "n_components"
    cdef int check_convergence = termination == "convergence"
    cdef int has_callback = callback is not None
    cdef int select_method = get_select_method(selection)
    cdef int permute = selection == "permute"
    cdef int stop = 0
    cdef int n_sv = 0


    for t in xrange(max_iter):
        if verbose >= 1:
            print "\nIteration", t

        Dpmax = 0

        if permute:
            rs.shuffle(index)

        for s in xrange(n_features):
            if permute:
                j = index[s]
            else:
                j = select_sv_precomputed(index, search_size,
                                         n_features, select_method, b, rs)

            X.get_column_ptr(j, &indices, &data, &n_nz)

            loss.solve_l2(j,
                          C,
                          <double*>w.data,
                          indices, data, n_nz,
                          <double*>col.data,
                          <double*>y.data,
                          <double*>b.data,
                          &Dp)

            if fabs(Dp) > Dpmax:
                Dpmax = fabs(Dp)

            if w[j] != 0:
                n_sv += 1

            # Exit if necessary.
            if check_n_sv and n_sv == n_components:
                stop = 1
                break

            # Callback
            if has_callback and s % 100 == 0:
                ret = callback(self)
                if ret is not None:
                    stop = 1
                    break

            if verbose >= 1 and s % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

        # end for (iterate over features)

        if stop:
            break

        if check_convergence and Dpmax < tol:
            if verbose >= 1:
                print "\nConverged at iteration", t
            break

    # for iterations

    if verbose >= 1:
        print

    return w


cpdef _C_lower_bound_kernel(np.ndarray[double, ndim=2, mode='c'] X,
                            np.ndarray[double, ndim=2, mode='c'] Y,
                            Kernel kernel,
                            search_size=None,
                            random_state=None):

    cdef int n_samples = X.shape[0]
    cdef int n = n_samples

    cdef int i, j, k, l
    cdef int n_vectors = Y.shape[1]

    cdef double val, max_ = -DBL_MAX

    cdef np.ndarray[int, ndim=1, mode='c'] ind
    ind = np.arange(n, dtype=np.int32)

    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n, dtype=np.float64)

    if search_size is not None:
        n = search_size
        random_state.shuffle(ind)

    for j in xrange(n):
        k = ind[j]

        for i in xrange(n_samples):
            col[i] = kernel.compute(X, i, X, k)

        for l in xrange(n_vectors):
            val = 0
            for i in xrange(n_samples):
                val += Y[i, l] * col[i]
            max_ = max(max_, fabs(val))

    return max_
