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

cdef extern from "math.h":
   double fabs(double)
   double exp(double x)
   double log(double x)
   double sqrt(double x)

cdef extern from "float.h":
   double DBL_MAX

cdef class LossFunction:

    # L2 regularization

    cdef void solve_l2(self,
                       int j,
                       int n_samples,
                       double C,
                       double *w,
                       double *col_ro,
                       double *col,
                       double *y,
                       double *b,
                       double *Dp):

        cdef double sigma = 0.01
        cdef double beta = 0.5
        cdef double bound, Dpp, Dj_zero, z, d

        self.derivatives_l2(j,
                            n_samples,
                            C,
                            sigma,
                            w,
                            col_ro,
                            col,
                            y,
                            b,
                            Dp,
                            &Dpp,
                            &Dj_zero,
                            &bound)

        if fabs(Dp[0]/Dpp) <= 1e-12:
            return

        d = -Dp[0] / Dpp

        z = self.line_search_l2(j,
                                n_samples,
                                d,
                                C,
                                sigma,
                                beta,
                                w,
                                col_ro,
                                col,
                                y,
                                b,
                                Dp[0],
                                Dpp,
                                Dj_zero,
                                bound)

        w[j] += z

    cdef void derivatives_l2(self,
                             int j,
                             int n_samples,
                             double C,
                             double sigma,
                             double *w,
                             double *col_ro,
                             double *col,
                             double *y,
                             double *b,
                             double *Dp,
                             double *Dpp,
                             double *Dj_zero,
                             double *bound):
        raise NotImplementedError()

    cdef double line_search_l2(self,
                               int j,
                               int n_samples,
                               double d,
                               double C,
                               double sigma,
                               double beta,
                               double *w,
                               double *col_ro,
                               double *col,
                               double *y,
                               double *b,
                               double Dp,
                               double Dpp,
                               double Dj_zero,
                               double bound):
        raise NotImplementedError()

    # L1/L2 regularization (multiclass)

    cdef void solve_l1l2_mc(self,
                            int j,
                            int n_samples,
                            int n_vectors,
                            double C,
                            np.ndarray[double, ndim=2, mode='c'] w,
                            double *col_ro,
                            np.ndarray[int, ndim=1] y,
                            np.ndarray[double, ndim=2, mode='c'] b,
                            np.ndarray[double, ndim=1, mode='c'] g,
                            np.ndarray[double, ndim=1, mode='c'] d,
                            np.ndarray[double, ndim=1, mode='c'] d_old,
                            np.ndarray[double, ndim=1, mode='c'] Z):

        cdef int i, k
        cdef double scaling, delta, L, R_j, Lpp_max

        cdef int max_num_linesearch = 20
        cdef double beta = 0.5
        cdef double sigma = 0.01

        self.derivatives_l1l2_mc(j, n_samples, n_vectors, C,
                                 w, col_ro, y, b, g, Z,
                                 &L, &R_j, &Lpp_max)


        # Compute vector to be projected.
        for k in xrange(n_vectors):
            d[k] = w[k, j] - g[k] / Lpp_max

        # Project.
        scaling = 0
        for k in xrange(n_vectors):
            scaling += d[k] * d[k]

        scaling = 1 - 1 / (Lpp_max * sqrt(scaling))

        if scaling < 0:
            scaling = 0

        delta = 0
        for k in xrange(n_vectors):
            d_old[k] = 0
            # Difference between new and old solution.
            d[k] = scaling * d[k] - w[k, j]
            delta += d[k] * g[k]

        # Perform line search.
        self.line_search_l1l2_mc(j, n_samples, n_vectors, C,
                                 w, col_ro, y, b, g, d, d_old, Z,
                                 L, R_j, &delta,
                                 max_num_linesearch, sigma, beta)

        # Update solution
        for k in xrange(n_vectors):
            w[k, j] += d[k]

    cdef void derivatives_l1l2_mc(self,
                                  int j,
                                  int n_samples,
                                  int n_vectors,
                                  double C,
                                  np.ndarray[double, ndim=2, mode='c'] w,
                                  double *col_ro,
                                  np.ndarray[int, ndim=1] y,
                                  np.ndarray[double, ndim=2, mode='c'] b,
                                  np.ndarray[double, ndim=1, mode='c'] g,
                                  np.ndarray[double, ndim=1, mode='c'] Z,
                                  double* L,
                                  double* R_j,
                                  double* Lpp_max):
        raise NotImplementedError

    cdef void line_search_l1l2_mc(self,
                                  int j,
                                  int n_samples,
                                  int n_vectors,
                                  double C,
                                  np.ndarray[double, ndim=2, mode='c'] w,
                                  double *col_ro,
                                  np.ndarray[int, ndim=1] y,
                                  np.ndarray[double, ndim=2, mode='c'] b,
                                  np.ndarray[double, ndim=1, mode='c'] g,
                                  np.ndarray[double, ndim=1, mode='c'] d,
                                  np.ndarray[double, ndim=1, mode='c'] d_old,
                                  np.ndarray[double, ndim=1, mode='c'] Z,
                                  double L,
                                  double R_j,
                                  double* delta,
                                  int max_num_linesearch,
                                  double sigma,
                                  double beta):
        raise NotImplementedError


cdef class Squared(LossFunction):


    cdef void solve_l2(self,
                       int j,
                       int n_samples,
                       double C,
                       double *w,
                       double *col_ro,
                       double *col,
                       double *y,
                       double *b,
                       double *Dp):
        cdef int i
        cdef double pred, num, denom, old_w, val, z

        num = 0
        denom = 0
        Dp[0] = 0

        for i in xrange(n_samples):
            val = col_ro[i] * y[i]
            col[i] = val

            Dp[0] -= b[i] * val
            pred = (1 - b[i]) * y[i]
            denom += col_ro[i] * col_ro[i]
            num += (y[i] - pred) * col_ro[i]

        denom *= 2 * C
        denom += 1
        num *= 2 * C

        Dp[0] = w[j] + 2 * C * Dp[0]
        num -= w[j]

        old_w = w[j]
        z = num/denom
        w[j] += z

        for i in xrange(n_samples):
            b[i] -= z * col[i]


cdef class SquaredHinge(LossFunction):

    # L2 regularization

    cdef void derivatives_l2(self,
                             int j,
                             int n_samples,
                             double C,
                             double sigma,
                             double *w,
                             double *col_ro,
                             double *col,
                             double *y,
                             double *b,
                             double *Dp,
                             double *Dpp,
                             double *Dj_zero,
                             double *bound):
        cdef int i
        cdef double xj_sq = 0
        cdef double val

        Dp[0] = 0
        Dpp[0] = 0
        Dj_zero[0] = 0

        for i in xrange(n_samples):
            val = col_ro[i] * y[i]
            col[i] = val
            xj_sq += val * val

            if b[i] > 0:
                Dp[0] -= b[i] * val
                Dpp[0] += val * val
                Dj_zero[0] += b[i] * b[i]

        Dp[0] = w[j] + 2 * C * Dp[0]
        Dpp[0] = 1 + 2 * C * Dpp[0]
        bound[0] = (2 * C * xj_sq + 1) / 2.0 + sigma

        Dj_zero[0] *= C

    cdef double line_search_l2(self,
                               int j,
                               int n_samples,
                               double d,
                               double C,
                               double sigma,
                               double beta,
                               double *w,
                               double *col_ro,
                               double *col,
                               double *y,
                               double *b,
                               double Dp,
                               double Dpp,
                               double Dj_zero,
                               double bound):
        cdef int step
        cdef double z_diff, z_old, z, Dj_z, b_new, cond

        z_old = 0
        z = d

        for step in xrange(100):
            z_diff = z_old - z

            # lambda <= Dpp/bound is equivalent to Dp/z <= -bound
            if Dp/z + bound <= 0:
                for i in xrange(n_samples):
                    b[i] += z_diff * col[i]
                break

            Dj_z = 0

            for i in xrange(n_samples):
                b_new = b[i] + z_diff * col[i]
                b[i] = b_new
                if b_new > 0:
                    Dj_z += b_new * b_new

            Dj_z *= C

            z_old = z

            #   0.5 * (w + z e_j)^T (w + z e_j)
            # = 0.5 * w^T w + w_j z + 0.5 z^2
            cond = w[j] * z + (0.5 + sigma) * z * z

            cond += Dj_z - Dj_zero

            if cond <= 0:
                break
            else:
                z *= beta

        return z

    # L1/L2 regularization (multi-class)

    cdef void derivatives_l1l2_mc(self,
                                  int j,
                                  int n_samples,
                                  int n_vectors,
                                  double C,
                                  np.ndarray[double, ndim=2, mode='c'] w,
                                  double *col_ro,
                                  np.ndarray[int, ndim=1] y,
                                  np.ndarray[double, ndim=2, mode='c'] b,
                                  np.ndarray[double, ndim=1, mode='c'] g,
                                  np.ndarray[double, ndim=1, mode='c'] Z,
                                  double* L,
                                  double* R_j,
                                  double* Lpp_max):

        cdef int i, k
        cdef double tmp

        # Compute objective value, gradient and largest second derivative.
        Lpp_max[0] = 0
        R_j[0] = 0
        L[0] = 0

        for k in xrange(n_vectors):
            g[k] = 0
            R_j[0] += w[k, j] * w[k, j]

            for i in xrange(n_samples):
                if y[i] == k:
                    continue

                if b[k, i] > 0:
                    L[0] += b[k, i] * b[k, i]
                    tmp = b[k, i] * col_ro[i]
                    g[y[i]] -= tmp
                    g[k] += tmp
                    Lpp_max[0] += col_ro[i] * col_ro[i]

        for k in xrange(n_vectors):
            g[k] *= 2 * C

        L[0] *= C
        Lpp_max[0] *= 2 * C
        Lpp_max[0] = min(max(Lpp_max[0], 1e-9), 1e9)
        R_j[0] = sqrt(R_j[0])

    cdef void line_search_l1l2_mc(self,
                                  int j,
                                  int n_samples,
                                  int n_vectors,
                                  double C,
                                  np.ndarray[double, ndim=2, mode='c'] w,
                                  double *col_ro,
                                  np.ndarray[int, ndim=1] y,
                                  np.ndarray[double, ndim=2, mode='c'] b,
                                  np.ndarray[double, ndim=1, mode='c'] g,
                                  np.ndarray[double, ndim=1, mode='c'] d,
                                  np.ndarray[double, ndim=1, mode='c'] d_old,
                                  np.ndarray[double, ndim=1, mode='c'] Z,
                                  double L,
                                  double R_j,
                                  double* delta,
                                  int max_num_linesearch,
                                  double sigma,
                                  double beta):

        cdef int i, k, n
        cdef double tmp, L_new, R_j_new, b_new

        for n in xrange(max_num_linesearch):

            # Update predictions, normalizations and objective value.
            L_new = 0
            for i in xrange(n_samples):
                tmp = d_old[y[i]] - d[y[i]]

                for k in xrange(n_vectors):
                    if k == y[i]:
                        continue

                    b_new = b[k, i] + (tmp - (d_old[k] - d[k])) * col_ro[i]
                    b[k, i] = b_new
                    if b_new > 0:
                        L_new += b_new * b_new

            L_new *= C

            # Compute regularization term.
            R_j_new = 0
            for k in xrange(n_vectors):
                tmp = w[k, j] + d[k]
                R_j_new += tmp * tmp
            R_j_new = sqrt(R_j_new)
            # R_new = R - R_j + R_j_new

            if n == 0:
                delta[0] += R_j_new - R_j
                delta[0] *= sigma

            # Check decrease condition
            if L_new - L + R_j_new - R_j <= delta[0]:
                break
            else:
                delta[0] *= beta
                for k in xrange(n_vectors):
                    d_old[k] = d[k]
                    d[k] *= beta


cdef class ModifiedHuber(LossFunction):

    cdef void derivatives_l2(self,
                             int j,
                             int n_samples,
                             double C,
                             double sigma,
                             double *w,
                             double *col_ro,
                             double *col,
                             double *y,
                             double *b,
                             double *Dp,
                             double *Dpp,
                             double *Dj_zero,
                             double *bound):
        cdef int i
        cdef double xj_sq = 0
        cdef double val

        Dp[0] = 0
        Dpp[0] = 0
        Dj_zero[0] = 0

        for i in xrange(n_samples):
            val = col_ro[i] * y[i]
            col[i] = val
            xj_sq += val * val

            if b[i] > 2:
                Dp[0] -= 2 * val
                # -4 yp = 4 (b[i] - 1)
                Dj_zero[0] += 4 * (b[i] - 1)
            elif b[i] > 0:
                Dp[0] -= b[i] * val
                Dpp[0] += val * val
                Dj_zero[0] += b[i] * b[i]

        Dp[0] = w[j] + 2 * C * Dp[0]
        Dpp[0] = 1 + 2 * C * Dpp[0]
        bound[0] = (2 * C * xj_sq + 1) / 2.0 + sigma

        Dj_zero[0] *= C


    cdef double line_search_l2(self,
                               int j,
                               int n_samples,
                               double d,
                               double C,
                               double sigma,
                               double beta,
                               double *w,
                               double *col_ro,
                               double *col,
                               double *y,
                               double *b,
                               double Dp,
                               double Dpp,
                               double Dj_zero,
                               double bound):
        cdef int step
        cdef double z_diff, z_old, z, Dj_z, b_new, cond

        z_old = 0
        z = d

        for step in xrange(100):
            z_diff = z_old - z

            # lambda <= Dpp/bound is equivalent to Dp/z <= -bound
            if Dp/z + bound <= 0:
                for i in xrange(n_samples):
                    b[i] += z_diff * col[i]
                break

            Dj_z = 0

            for i in xrange(n_samples):
                b_new = b[i] + z_diff * col[i]
                b[i] = b_new

                if b_new > 2:
                    Dj_z += 4 * (b[i] - 1)
                elif b_new > 0:
                    Dj_z += b_new * b_new

            Dj_z *= C

            z_old = z

            #   0.5 * (w + z e_j)^T (w + z e_j)
            # = 0.5 * w^T w + w_j z + 0.5 z^2
            cond = w[j] * z + (0.5 + sigma) * z * z

            cond += Dj_z - Dj_zero

            if cond <= 0:
                break
            else:
                z *= beta

        return z


cdef class Log(LossFunction):

    # L2 regularization

    cdef void derivatives_l2(self,
                             int j,
                             int n_samples,
                             double C,
                             double sigma,
                             double *w,
                             double *col_ro,
                             double *col,
                             double *y,
                             double *b,
                             double *Dp,
                             double *Dpp,
                             double *Dj_zero,
                             double *bound):
        cdef int i
        cdef double xj_sq = 0
        cdef double val, tau, exppred

        Dp[0] = 0
        Dpp[0] = 0
        Dj_zero[0] = 0

        for i in xrange(n_samples):
            val = col_ro[i] * y[i]
            col[i] = val

            exppred = 1 + 1 / b[i]
            tau = 1 / exppred
            Dp[0] += val * (tau - 1)
            Dpp[0] += val * val * tau * (1 - tau)
            Dj_zero[0] += log(exppred)

        Dp[0] = w[j] + C * Dp[0]
        Dpp[0] = 1 + C * Dpp[0]

        Dj_zero[0] *= C


    cdef double line_search_l2(self,
                               int j,
                               int n_samples,
                               double d,
                               double C,
                               double sigma,
                               double beta,
                               double *w,
                               double *col_ro,
                               double *col,
                               double *y,
                               double *b,
                               double Dp,
                               double Dpp,
                               double Dj_zero,
                               double bound):
        cdef int step
        cdef double z_diff, z_old, z, Dj_z, exppred, cond

        z_old = 0
        z = d

        for step in xrange(100):
            z_diff = z - z_old
            Dj_z = 0

            for i in xrange(n_samples):
                b[i] *= exp(z_diff * col[i])
                exppred = 1 + 1 / b[i]
                Dj_z += log(exppred)

            Dj_z *= C

            z_old = z

            #   0.5 * (w + z e_j)^T (w + z e_j)
            # = 0.5 * w^T w + w_j z + 0.5 z^2
            cond = w[j] * z + (0.5 + sigma) * z * z

            cond += Dj_z - Dj_zero

            if cond <= 0:
                break
            else:
                z *= beta

        return z

    # L1/L2 regulariztion (multiclass)

    cdef void derivatives_l1l2_mc(self,
                                  int j,
                                  int n_samples,
                                  int n_vectors,
                                  double C,
                                  np.ndarray[double, ndim=2, mode='c'] w,
                                  double *col_ro,
                                  np.ndarray[int, ndim=1] y,
                                  np.ndarray[double, ndim=2, mode='c'] b,
                                  np.ndarray[double, ndim=1, mode='c'] g,
                                  np.ndarray[double, ndim=1, mode='c'] Z,
                                  double* L,
                                  double* R_j,
                                  double* Lpp_max):

        cdef int i, k
        cdef double Lpp, tmp

        # Compute normalization and objective value.
        L[0] = 0
        for i in xrange(n_samples):
            Z[i] = 0
            for k in xrange(n_vectors):
                Z[i] += b[k, i]
            L[0] += log(Z[i])
        L[0] *= C

        # Compute gradient and largest second derivative.
        Lpp_max[0] = -DBL_MAX
        R_j[0] = 0

        for k in xrange(n_vectors):
            g[k] = 0
            R_j[0] += w[k, j] * w[k, j]

            Lpp = 0

            for i in xrange(n_samples):
                if y[i] == k:
                    continue

                if Z[i] > 0:
                    tmp = b[k, i]
                    g[k] += tmp * col_ro[i] / Z[i]
                    Lpp += col_ro[i] * col_ro[i] * tmp * (1 - tmp/Z[i]) / Z[i]

            g[k] *= C
            Lpp *= C
            Lpp_max[0] = max(Lpp, Lpp_max[0])

        Lpp_max[0] = min(max(Lpp_max[0], 1e-9), 1e9)
        R_j[0] = sqrt(R_j[0])

    cdef void line_search_l1l2_mc(self,
                                  int j,
                                  int n_samples,
                                  int n_vectors,
                                  double C,
                                  np.ndarray[double, ndim=2, mode='c'] w,
                                  double *col_ro,
                                  np.ndarray[int, ndim=1] y,
                                  np.ndarray[double, ndim=2, mode='c'] b,
                                  np.ndarray[double, ndim=1, mode='c'] g,
                                  np.ndarray[double, ndim=1, mode='c'] d,
                                  np.ndarray[double, ndim=1, mode='c'] d_old,
                                  np.ndarray[double, ndim=1, mode='c'] Z,
                                  double L,
                                  double R_j,
                                  double* delta,
                                  int max_num_linesearch,
                                  double sigma,
                                  double beta):

        cdef int i, k, n
        cdef double tmp, L_new, R_j_new

        for n in xrange(max_num_linesearch):

            # Update predictions, normalizations and objective value.
            L_new = 0
            for i in xrange(n_samples):
                tmp = d_old[y[i]] - d[y[i]]
                Z[i] = 0

                for k in xrange(n_vectors):
                    b[k, i] *= exp((d[k] - d_old[k] + tmp) * col_ro[i])
                    Z[i] += b[k, i]

                L_new += log(Z[i])

            L_new *= C

            # Compute regularization term.
            R_j_new = 0
            for k in xrange(n_vectors):
                tmp = w[k, j] + d[k]
                R_j_new += tmp * tmp
            R_j_new = sqrt(R_j_new)
            # R_new = R - R_j + R_j_new

            if n == 0:
                delta[0] += R_j_new - R_j
                delta[0] *= sigma

            # Check decrease condition
            if L_new - L + R_j_new - R_j <= delta[0]:
                break
            else:
                delta[0] *= beta
                for k in xrange(n_vectors):
                    d_old[k] = d[k]
                    d[k] *= beta


def _primal_cd_l2svm_l1r(self,
                         np.ndarray[double, ndim=1, mode='c'] w,
                         np.ndarray[double, ndim=1, mode='c'] b,
                         X,
                         np.ndarray[double, ndim=1] y,
                         np.ndarray[int, ndim=1, mode='c'] index,
                         KernelCache kcache,
                         int linear_kernel,
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

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]

    cdef np.ndarray[double, ndim=2, mode='fortran'] Xf
    cdef np.ndarray[double, ndim=2, mode='c'] Xc

    if linear_kernel:
        Xf = X
    else:
        Xc = X
        n_features = n_samples

    cdef int j, s, t, i = 0
    cdef int active_size = n_features
    cdef int max_num_linesearch = 20

    cdef double sigma = 0.01
    cdef double beta = 0.5
    cdef double d, Lp, Lpp, Lpp_wj
    cdef double Lpmax_old = DBL_MAX
    cdef double Lpmax_new
    cdef double Lpmax_init
    cdef double z, z_old, z_diff
    cdef double Lj_zero, Lj_z
    cdef double appxcond, cond
    cdef double val, val_sq
    cdef double Lp_p, Lp_n, violation
    cdef double delta, b_new, b_add
    cdef double xj_sq
    cdef double wj_abs

    cdef double* col_data
    cdef double* col_ro
    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)
    col_data = <double*>col.data

    cdef list[int].iterator it

    cdef int check_n_sv = termination == "n_components"
    cdef int check_convergence = termination == "convergence"
    cdef int stop = 0
    cdef int select_method = get_select_method(selection)
    cdef int permute = selection == "permute" or linear_kernel
    cdef int has_callback = callback is not None

    # FIXME: would be better to store the support indices in the class.
    if not linear_kernel:
        for j in xrange(n_features):
            if w[j] != 0:
                kcache.add_sv(j)

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
                                          active_size, select_method, b, kcache,
                                          0, rs)

            Lj_zero = 0
            Lp = 0
            Lpp = 0
            xj_sq = 0

            if linear_kernel:
                col_ro = (<double*>Xf.data) + j * n_samples
            else:
                kcache.compute_column(Xc, Xc, j, col)
                col_ro = col_data

            for i in xrange(n_samples):
                val = col_ro[i] * y[i]
                col[i] = val
                val_sq = val * val
                if b[i] > 0:
                    Lp -= val * b[i]
                    Lpp += val_sq
                    Lj_zero += b[i] * b[i]
                xj_sq += val_sq
            # end for

            xj_sq *= C
            Lj_zero *= C
            Lp *= 2 * C

            Lpp *= 2 * C
            Lpp = max(Lpp, 1e-12)

            Lp_p = Lp + 1
            Lp_n = Lp - 1
            violation = 0

            # Shrinking.
            if w[j] == 0:
                if Lp_p < 0:
                    violation = -Lp_p
                elif Lp_n > 0:
                    violation = Lp_n
                elif Lp_p > Lpmax_old / n_samples and Lp_n < -Lpmax_old / n_samples:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    # Jump w/o incrementing s so as to use the swapped sample.
                    continue
            elif w[j] > 0:
                violation = fabs(Lp_p)
            else:
                violation = fabs(Lp_n)

            Lpmax_new = max(Lpmax_new, violation)

            # Obtain Newton direction d.
            Lpp_wj = Lpp * w[j]
            if Lp_p <= Lpp_wj:
                d = -Lp_p / Lpp
            elif Lp_n >= Lpp_wj:
                d = -Lp_n / Lpp
            else:
                d = -w[j]

            if fabs(d) < 1.0e-12:
                s += 1
                continue

            wj_abs = fabs(w[j])
            delta = fabs(w[j] + d) - wj_abs + Lp * d
            z_old = 0
            z = d

            # Check z = lambda*d for lambda = 1, beta, beta^2 such that
            # sufficient decrease condition is met.
            for num_linesearch in xrange(max_num_linesearch):
                # Reversed because of the minus in b[i] = 1 - y_i w^T x_i.
                z_diff = z_old - z
                cond = fabs(w[j] + z) - wj_abs - sigma * delta

                appxcond = xj_sq * z * z + Lp * z + cond

                # Avoid line search if possible.
                if appxcond <= 0:
                    for i in xrange(n_samples):
                        # Need to remove the old z and had the new one.
                        b[i] += z_diff * col[i]
                    break

                # Compute objective function value.
                Lj_z = 0

                for i in xrange(n_samples):
                    b_new = b[i] + z_diff * col[i]
                    b[i] = b_new
                    if b_new > 0:
                        Lj_z += b_new * b_new

                Lj_z *= C

                # Check stopping condition.
                cond = cond + Lj_z - Lj_zero
                if cond <= 0:
                    break
                else:
                    z_old = z
                    z *= beta
                    delta *= beta

            # end for num_linesearch

            w[j] += z

            # Update support set.
            if not linear_kernel:
                if w[j] != 0:
                    kcache.add_sv(j)
                elif w[j] == 0:
                    kcache.remove_sv(j)

            # Exit if necessary.
            if check_n_sv and kcache.n_sv() >= n_components:
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
                     X,
                     np.ndarray[int, ndim=1] y,
                     np.ndarray[int, ndim=1, mode='c'] index,
                     LossFunction loss,
                     KernelCache kcache,
                     int linear_kernel,
                     double C,
                     int max_iter,
                     RandomState rs,
                     double tol,
                     callback,
                     int verbose):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]
    cdef Py_ssize_t n_vectors = w.shape[0]

    cdef np.ndarray[double, ndim=2, mode='fortran'] Xf
    cdef np.ndarray[double, ndim=2, mode='c'] Xc

    if linear_kernel:
        Xf = X
    else:
        Xc = X
        n_features = index.shape[0]

    cdef int t, s, i, j, k, n
    cdef int active_size = n_features

    cdef double* col_data
    cdef double* col_ro
    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)
    col_data = <double*>col.data

    cdef np.ndarray[double, ndim=1, mode='c'] g
    g = np.zeros(n_vectors, dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] d
    d = np.zeros(n_vectors, dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] d_old
    d_old = np.zeros(n_vectors, dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] Z
    Z = np.zeros(n_samples, dtype=np.float64)

    for t in xrange(max_iter):
        if verbose >= 1:
            print "\nIteration", t

        rs.shuffle(index[:active_size])

        for s in xrange(active_size):
            j = index[s]

            if linear_kernel:
                col_ro = (<double*>Xf.data) + j * n_samples
            else:
                kcache.compute_column(Xc, Xc, j, col)
                col_ro = col_data

            loss.solve_l1l2_mc(j, n_samples, n_vectors, C,
                               w, col_ro, y, b, g, d, d_old, Z)


def _primal_cd_l2r(self,
                   np.ndarray[double, ndim=1, mode='c'] w,
                   np.ndarray[double, ndim=1, mode='c'] b,
                   X,
                   A,
                   np.ndarray[double, ndim=1] y,
                   np.ndarray[int, ndim=1, mode='c'] index,
                   LossFunction loss,
                   KernelCache kcache,
                   int linear_kernel,
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

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]

    cdef np.ndarray[double, ndim=2, mode='fortran'] Xf
    cdef np.ndarray[double, ndim=2, mode='c'] Xc
    cdef np.ndarray[double, ndim=2, mode='c'] Ac

    if linear_kernel:
        Xf = X
    else:
        Xc = X
        Ac = A
        n_features = index.shape[0]

    cdef int i, j, s, t
    cdef double Dp, Dpmax

    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)
    # Container to store X[i, j] * y[i] for all i.
    cdef double *col_ptr = <double*>col.data

    cdef np.ndarray[double, ndim=1, mode='c'] col_ro
    col_ro = np.zeros(n_samples, dtype=np.float64)
    # Pointer to X[i, j] for all i (read-only).
    cdef double *col_ro_ptr = <double*>col_ro.data

    cdef int check_n_sv = termination == "n_components"
    cdef int check_convergence = termination == "convergence"
    cdef int has_callback = callback is not None
    cdef int select_method = get_select_method(selection)
    cdef int permute = selection == "permute" or linear_kernel
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
                                          n_features, select_method, b, kcache,
                                          0, rs)

            if linear_kernel:
                col_ro_ptr = (<double*>Xf.data) + j * n_samples
            else:
                kcache.compute_column(Xc, Ac, j, col_ro)

            loss.solve_l2(j,
                          n_samples,
                          C,
                          <double*>w.data,
                          col_ro_ptr,
                          col_ptr,
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
