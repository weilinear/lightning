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

from lightning.select_fast cimport get_select_method
from lightning.select_fast cimport select_sv_precomputed
from lightning.random.random_fast cimport RandomState
from lightning.dataset_fast cimport Dataset
from lightning.dataset_fast cimport KernelDataset

DEF LOWER = 1e-2
DEF UPPER = 1e9

cdef extern from "math.h":
   double fabs(double)
   double exp(double x)
   double log(double x)
   double sqrt(double x)

cdef extern from "float.h":
   double DBL_MAX

cdef class LossFunction:

    cdef int max_steps
    cdef double sigma
    cdef double beta
    cdef int verbose

    # L2 regularization

    cdef void solve_l2(self,
                       int j,
                       double C,
                       double alpha,
                       double *w,
                       int *indices,
                       double *data,
                       int n_nz,
                       double *y,
                       double *b,
                       double *Dp):

        cdef double Dpp, Dj_zero, z, d
        cdef int i, ii, step
        cdef double z_diff, z_old, Dj_z, cond

        # Compute derivatives
        self.derivatives(j, C, indices, data, n_nz, y, b,
                         Dp, &Dpp, &Dj_zero)

        Dp[0] = alpha * w[j] + Dp[0]
        Dpp = alpha + Dpp

        if fabs(Dp[0]/Dpp) <= 1e-12:
            return

        d = -Dp[0] / Dpp

        # Perform line search
        z_old = 0
        z = d

        step = 1
        while True:
            z_diff = z_old - z

            # Update old predictions
            self.update(j, z_diff, C, indices, data, n_nz,
                        y, b, &Dj_z)

            if step == self.max_steps:
                if self.verbose >= 3 and self.max_steps > 1:
                    print "Max steps reached during line search..."
                break

            #   0.5 * alpha * (w + z e_j)^T (w + z e_j)
            # = 0.5 * alpha * w^T w + alpha * w_j z + 0.5 * alpha * z^2
            cond = alpha * w[j] * z + (0.5 * alpha + self.sigma) * z * z
            cond += Dj_z - Dj_zero
            if cond <= 0:
                break

            z_old = z
            z *= self.beta
            step += 1

        w[j] += z

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L):
        raise NotImplementedError()

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
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
                      double *y,
                      double *b,
                      double violation_old,
                      double *violation,
                      int *n_sv,
                      int shrinking):
        cdef double Lj_zero = 0
        cdef double Lp = 0
        cdef double Lpp = 0
        cdef double xj_sq = 0
        cdef double Lpp_wj, d, wj_abs
        cdef double cond
        cdef double appxcond = 0
        cdef double Lj_z
        cdef int step

        # Compute derivatives
        self.derivatives(j, C, indices, data, n_nz, y, b,
                         &Lp, &Lpp, &Lj_zero)

        Lpp = max(Lpp, 1e-12)

        Lp_p = Lp + 1
        Lp_n = Lp - 1
        violation[0] = 0

        # Violation and shrinking.
        if w[j] == 0:
            if Lp_p < 0:
                violation[0] = -Lp_p
            elif Lp_n > 0:
                violation[0] = Lp_n
            elif shrinking and \
                 Lp_p > violation_old / n_samples and \
                 Lp_n < -violation_old / n_samples:
                # Shrink!
                if self.verbose >= 4:
                    print "Shrink variable", j
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
            cond = fabs(w[j] + z) - wj_abs - self.sigma * delta

            # Compute objective function value.
            self.update(j, z_diff, C, indices, data, n_nz, y, b, &Lj_z)

            if step == self.max_steps:
                if self.verbose >= 3 and self.max_steps > 1:
                    print "Max steps reached during line search..."
                break

            # Check stopping condition.
            cond = cond + Lj_z - Lj_zero
            if cond <= 0:
                break

            z_old = z
            z *= self.beta
            delta *= self.beta
            step += 1

        # end for num_linesearch

        if w[j] == 0 and z != 0:
            n_sv[0] += 1
        elif z != 0 and w[j] == -z:
            n_sv[0] -= 1

        # Update w.
        w[j] += z

        # If recompute b[], need to do it here.

        return 0

    # L1/L2 regularization

    cdef int solve_l1l2(self,
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
                        double* Z,
                        double violation_old,
                        double *violation,
                        int shrinking):

        cdef int n_samples = Y.shape[0]
        cdef int i, k, ii, step
        cdef double scaling, delta, L, R_j, Lpp_max, dmax
        cdef double tmp, L_new, R_j_new
        cdef double L_tmp, Lpp_tmp
        cdef double* y_ptr
        cdef double* b_ptr
        cdef double z_diff, g_norm
        cdef double lmbda = 1.0
        cdef int nv = n_samples * n_vectors

        # Compute partial gradient.
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
                self.derivatives(j, C, indices, data, n_nz, y_ptr,
                                 b_ptr, &g[k], &Lpp_tmp, &L_tmp)
                L += L_tmp
                Lpp_max = max(Lpp_max, Lpp_tmp)
                y_ptr += n_samples
                b_ptr += n_samples
                Z_ptr += n_samples

            Lpp_max = min(max(Lpp_max, LOWER), UPPER)

        # Compute partial gradient norm and regularization term.
        g_norm = 0
        R_j = 0

        for k in xrange(n_vectors):
            g_norm += g[k] * g[k]
            R_j += w[k, j] * w[k, j]

        g_norm = sqrt(g_norm)
        R_j = sqrt(R_j)

        # Violation and shrinking.
        if R_j == 0:
            g_norm -= lmbda
            if g_norm > 0:
                violation[0] = g_norm
            elif shrinking and \
                 g_norm + violation_old / nv <= 0:
                # Shrink!
                if self.verbose >= 4:
                    print "Shrink variable", j
                return 1
        else:
            violation[0] = fabs(g_norm - lmbda)

        # Compute vector to be projected.
        for k in xrange(n_vectors):
            d_old[k] = 0
            d[k] = w[k, j] - g[k] / Lpp_max

        # Project.
        scaling = 0
        for k in xrange(n_vectors):
            scaling += d[k] * d[k]

        scaling = 1 - lmbda / (Lpp_max * sqrt(scaling))

        if scaling < 0:
            scaling = 0

        delta = 0
        dmax = -DBL_MAX
        for k in xrange(n_vectors):
            # Difference between new and old solution.
            d[k] = scaling * d[k] - w[k, j]
            delta += d[k] * g[k]
            dmax = max(dmax, fabs(d[k]))

        # Check optimality.
        if dmax < 1e-12:
            return 0

        # Perform line search.
        step = 1
        while True:

            # Update predictions, normalizations and objective value.
            if multiclass:
                self.update_mc(C, n_vectors, indices, data, n_nz,
                               y, b, d, d_old, Z, &L_new)
            else:
                L_new = 0
                y_ptr = <double*>Y.data
                b_ptr = <double*>b.data

                for k in xrange(n_vectors):
                    z_diff = d_old[k] - d[k]
                    self.update(j, z_diff, C, indices, data, n_nz,
                                y_ptr, b_ptr, &L_tmp)
                    L_new += L_tmp
                    y_ptr += n_samples
                    b_ptr += n_samples
                    Z_ptr += n_samples

            if step == self.max_steps:
                if self.verbose >= 3 and self.max_steps > 1:
                    print "Max steps reached during line search..."
                break

            # Compute regularization term.
            R_j_new = 0
            for k in xrange(n_vectors):
                tmp = w[k, j] + d[k]
                R_j_new += tmp * tmp
            R_j_new = sqrt(R_j_new)
            # R_new = R - R_j + R_j_new

            if step == 1:
                delta += R_j_new - R_j
                delta *= self.sigma

            # Check decrease condition
            if L_new - L + R_j_new - R_j <= delta:
                break

            delta *= self.beta
            for k in xrange(n_vectors):
                d_old[k] = d[k]
                d[k] *= self.beta
            step += 1

        # Update solution
        for k in xrange(n_vectors):
            w[k, j] += d[k]

        return 0

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

    def __init__(self, verbose=0):
        self.max_steps = 1
        self.beta = 0.5
        self.sigma = 0.01
        self.verbose = verbose

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L):
        cdef int ii, i
        cdef double tmp

        Lp[0] = 0
        Lpp[0] = 0
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            tmp = data[ii] * C
            Lpp[0] += data[ii] * tmp
            Lp[0] += b[i] * tmp
            L[0] += C * b[i] * b[i]

        Lpp[0] *= 2
        Lp[0] *= 2

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int ii, i

        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b[i] -= z_diff * data[ii]
            L_new[0] += C * b[i] * b[i]


cdef class SquaredHinge(LossFunction):

    def __init__(self,
                 int max_steps=20,
                 double sigma=0.01,
                 double beta=0.5,
                 int verbose=0):
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.verbose = verbose

    # Binary

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L):
        cdef int i, ii
        cdef double val, tmp

        Lp[0] = 0
        Lpp[0] = 0
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            val = data[ii] * y[i]

            if b[i] > 0:
                tmp = val * C
                Lp[0] -= b[i] * tmp
                Lpp[0] += val * tmp
                L[0] += C * b[i] * b[i]

        Lp[0] *= 2
        Lpp[0] *= 2

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int i, ii
        cdef double b_new

        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b_new = b[i] + z_diff * data[ii] * y[i]
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
        cdef double tmp, tmp2

        # Compute objective value, gradient and largest second derivative.
        Lpp_max[0] = 0
        L[0] = 0

        for k in xrange(n_vectors):
            g[k] = 0
            Z[k] = 0

        for k in xrange(n_vectors):

            for ii in xrange(n_nz):
                i = indices[ii]

                if y[i] == k:
                    continue

                if b[k, i] > 0:
                    L[0] += C * b[k, i] * b[k, i]
                    tmp = C * data[ii]
                    tmp2 = tmp * b[k, i]
                    g[y[i]] -= tmp2
                    g[k] += tmp2
                    tmp2 = tmp * data[ii]
                    Z[y[i]] += tmp2
                    Z[k] += tmp2

        Lpp_max[0] = -DBL_MAX
        for k in xrange(n_vectors):
            g[k] *= 2
            Lpp_max[0] = max(Lpp_max[0], Z[k])

        Lpp_max[0] *= 2
        Lpp_max[0] = min(max(Lpp_max[0], LOWER), UPPER)

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
                    L_new[0] += C * b_new * b_new


cdef class ModifiedHuber(LossFunction):

    def __init__(self,
                 int max_steps=30,
                 double sigma=0.01,
                 double beta=0.5,
                 int verbose=0):
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.verbose = verbose

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L):
        cdef int i, ii
        cdef double val, tmp

        Lp[0] = 0
        Lpp[0] = 0
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            val = data[ii] * y[i]

            if b[i] > 2:
                Lp[0] -= 2 * val * C
                # -4 yp = 4 (b[i] - 1)
                L[0] += 4 * C * (b[i] - 1)
            elif b[i] > 0:
                tmp = val * C
                Lp[0] -= b[i] * tmp
                Lpp[0] += val * tmp
                L[0] += C * b[i] * b[i]

        Lp[0] *= 2
        Lpp[0] *= 2

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int i, ii
        cdef double b_new

        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b_new = b[i] + z_diff * data[ii] * y[i]
            b[i] = b_new

            if b_new > 2:
                L_new[0] += 4 * C * (b[i] - 1)
            elif b_new > 0:
                L_new[0] += C * b_new * b_new


cdef class Log(LossFunction):

    def __init__(self,
                 int max_steps=30,
                 double sigma=0.01,
                 double beta=0.5,
                 int verbose=0):
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.verbose = verbose

    # Binary

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L):
        cdef int i, ii
        cdef double val, tau, exppred, tmp

        Lp[0] = 0
        Lpp[0] = 0
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            val = data[ii] * y[i]

            exppred = 1 + 1 / b[i]
            tau = 1 / exppred
            tmp = val * C
            Lp[0] += tmp * (tau - 1)
            Lpp[0] += tmp * val * tau * (1 - tau)
            L[0] += C * log(exppred)


    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int i, ii
        cdef double exppred

        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b[i] /= exp(z_diff * data[ii] * y[i])
            exppred = 1 + 1 / b[i]
            L_new[0] += C * log(exppred)

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
        cdef double Lpp, tmp, tmp2

        # Compute normalization and objective value.
        L[0] = 0
        for ii in xrange(n_nz):
            i = indices[ii]
            Z[i] = 0
            for k in xrange(n_vectors):
                Z[i] += b[k, i]
            L[0] += C * log(Z[i])

        # Compute gradient and largest second derivative.
        Lpp_max[0] = -DBL_MAX

        for k in xrange(n_vectors):
            g[k] = 0
            Lpp = 0

            for ii in xrange(n_nz):
                i = indices[ii]

                if Z[i] == 0:
                    continue

                tmp = b[k, i] / Z[i]
                tmp2 = data[ii] * C
                Lpp += tmp2 * data[ii] * tmp * (1 - tmp)

                if k == y[i]:
                    tmp -= 1

                g[k] += tmp * tmp2

            Lpp_max[0] = max(Lpp, Lpp_max[0])

        Lpp_max[0] = min(max(Lpp_max[0], LOWER), UPPER)

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

            L_new[0] += C * log(Z[i])


def _primal_cd(self,
               np.ndarray[double, ndim=2, mode='c'] w,
               np.ndarray[double, ndim=2, mode='c'] b,
               Dataset X,
               np.ndarray[int, ndim=1] y,
               np.ndarray[double, ndim=2, mode='fortran'] Y,
               int k,
               int multiclass,
               np.ndarray[int, ndim=1, mode='c'] active_set,
               int penalty,
               LossFunction loss,
               selection,
               int search_size,
               termination,
               int n_components,
               double C,
               double alpha,
               int max_iter,
               int shrinking,
               double Gnorm_init,
               RandomState rs,
               double tol,
               callback,
               int n_calls,
               int verbose):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = active_set.shape[0]
    cdef int n_vectors = w.shape[0]

    # Initialization
    cdef int t, s, i, j, n
    cdef int active_size = n_features
    cdef double violation_old = DBL_MAX
    cdef double violation_new
    cdef double Gnorm_new
    cdef double Dpmax, Dp
    cdef double violation
    cdef int check_convergence = termination == "convergence"
    cdef int check_n_sv = termination == "n_components"
    cdef int select_method = get_select_method(selection)
    cdef int permute = selection == "permute"
    cdef int stop = 0
    cdef int has_callback = callback is not None
    cdef int shrink = 0
    cdef int* active_set_ptr = <int*>active_set.data
    cdef double* b_ptr
    cdef double* y_ptr
    cdef double* w_ptr
    cdef int n_sv = 0

    cdef np.ndarray[double, ndim=1, mode='c'] g
    cdef np.ndarray[double, ndim=1, mode='c'] d
    cdef np.ndarray[double, ndim=1, mode='c'] d_old
    cdef np.ndarray[double, ndim=2, mode='c'] buf

    if k == -1:
        g = np.zeros(n_vectors, dtype=np.float64)
        d = np.zeros(n_vectors, dtype=np.float64)
        d_old = np.zeros(n_vectors, dtype=np.float64)
        if multiclass:
            buf = np.zeros((n_samples, 1), dtype=np.float64)
        else:
            buf = np.zeros((n_vectors, n_samples), dtype=np.float64)

        b_ptr = <double*>b.data
    else:
        b_ptr = <double*>b.data + k * n_samples
        y_ptr = <double*>Y.data + k * n_samples
        w_ptr = <double*>w.data + k * n_features
        buf = np.zeros((n_samples, 1), dtype=np.float64)

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    for t in xrange(max_iter):
        if verbose >= 2:
            print "\nIteration", t

        rs.shuffle(active_set[:active_size])

        violation_new = 0
        Gnorm_new = 0
        Dpmax = 0

        s = 0
        while s < active_size:
            # Select coordinate.
            if permute:
                j = active_set[s]
            else:
                j = select_sv_precomputed(active_set_ptr, search_size,
                                          active_size, select_method, b_ptr, rs)

            # Retrieve column.
            X.get_column_ptr(j, &indices, &data, &n_nz)

            # Solve sub-problem.
            if penalty == 1:
                shrink = loss.solve_l1(j, C, w_ptr, n_samples,
                                       indices, data, n_nz,
                                       y_ptr, b_ptr, violation_old,
                                       &violation, &n_sv, shrinking)
            elif penalty == 12:
                shrink = loss.solve_l1l2(j, C, w, n_vectors,
                                         indices, data, n_nz,
                                         <int*>y.data, Y, multiclass,
                                         b, <double*>g.data, <double*>d.data,
                                         <double*>d_old.data, <double*>buf.data,
                                         violation_old, &violation, shrinking)
            elif penalty == 2:
                loss.solve_l2(j, C, alpha, w_ptr, indices, data, n_nz,
                              y_ptr, b_ptr, &Dp)
                Dpmax = max(Dpmax, fabs(Dp))
                if w_ptr[j] != 0:
                    n_sv += 1

            # Check if need to shrink.
            if shrink:
                active_size -= 1
                active_set[s], active_set[active_size] = \
                    active_set[active_size], active_set[s]
                continue

            # Update maximum absolute derivative.
            violation_new = max(violation_new, violation)
            Gnorm_new += violation

            # Exit if necessary.
            if check_n_sv and n_sv >= n_components:
                stop = 1
                break

            # Callback
            if has_callback and s % n_calls == 0:
                ret = callback(self)
                if ret is not None:
                    stop = 1
                    break

            # Output progress.
            if verbose >= 2 and s % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

            s += 1
        # end while active_size

        if stop:
            break

        if t == 0 and Gnorm_init == 0:
            Gnorm_init = Gnorm_new

        if verbose >= 2:
            print "\nActive size:", active_size

        # Check convergence.
        if check_convergence:
            if penalty == 2:
                if Dpmax < tol:
                    if verbose >= 1:
                        print "\nConverged at iteration", t
                    break
            else:
                if Gnorm_new <= tol * Gnorm_init:
                    if active_size == n_features:
                        if verbose >= 1:
                            print "\nConverged at iteration", t
                        break
                    else:
                        active_size = n_features
                        violation_old = DBL_MAX
                        continue

        violation_old = violation_new

    if verbose >= 1:
        print

    return Gnorm_init


cpdef _C_lower_bound_kernel(KernelDataset kds,
                            np.ndarray[double, ndim=2, mode='fortran'] Y,
                            search_size=None,
                            random_state=None):

    cdef int n_samples = kds.get_n_samples()
    cdef int n = n_samples

    cdef int i, j, k, l
    cdef int n_vectors = Y.shape[1]

    cdef double val, max_ = -DBL_MAX
    cdef int* indices
    cdef double* data
    cdef int n_nz

    cdef np.ndarray[int, ndim=1, mode='c'] ind
    ind = np.arange(n_samples, dtype=np.int32)

    if search_size is not None:
        n = search_size
        random_state.shuffle(ind)

    for j in xrange(n):
        k = ind[j]

        kds.get_column_ptr(k, &indices, &data, &n_nz)

        for l in xrange(n_vectors):
            val = 0
            for i in xrange(n_samples):
                val += Y[i, l] * data[i]
            max_ = max(max_, fabs(val))

    return max_
