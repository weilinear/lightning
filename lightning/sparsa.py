# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot

from .base import BaseClassifier

class Loss(object):
    pass

class SquaredHinge(Loss):

    def gradient(self, df, X, y):
        n_samples, n_features = X.shape
        n_vectors = df.shape[1]
        G = np.zeros((n_vectors, n_features), dtype=np.float64)
        for i in xrange(n_samples):
            for k in xrange(n_vectors):
                if y[i] == k:
                    continue
                update = max(1 - df[i, y[i]] + df[i, k], 0)
                if update != 0:
                    update = update * X[i]
                    G[y[i]] -= update
                    G[k] += update
        G *= 2
        return G


    def objective(self, df, y):
        n_samples = df.shape[0]
        ind = np.arange(n_samples)
        dfy = df[ind, y]
        obj = np.sum(np.maximum(1 - dfy[:, np.newaxis] + df, 0) ** 2)
        obj -= n_samples
        return obj


class Penalty(object):
    pass


class L1L2(Penalty):

    def projection(self, coef, alpha, L):
        n_features = coef.shape[1]
        l2norms = np.sqrt(np.sum(coef ** 2, axis=0))
        scales = np.maximum(1.0 - alpha / (L * l2norms), 0)
        coef *= scales

    def regularization(self, coef):
        return np.sum(np.sqrt(np.sum(coef ** 2, axis=0)))


class SparsaClassifier(BaseClassifier, ClassifierMixin):

    def __init__(self, C=1.0, alpha=1.0,
                 loss="squared_hinge", penalty="l1/l2", max_iter=100,
                 Lmin=1e-30, Lmax=1e30, L_factor=0.8,
                 max_steps=30, eta=2.0, sigma=1e-5,
                 verbose=0):
        self.C = C
        self.alpha = alpha
        self.loss = loss
        self.penalty = penalty
        self.max_iter = max_iter
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.L_factor = L_factor
        self.max_steps = max_steps
        self.eta = eta
        self.sigma = 1e-5
        self.verbose = verbose

    def _get_loss(self):
        losses = {
            "squared_hinge" : SquaredHinge(),
        }
        return losses[self.loss]

    def _get_penalty(self):
        penalties = {
            "l1/l2" : L1L2(),
        }
        return penalties[self.penalty]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y, n_classes, n_vectors = self._set_label_transformers(y, reencode=True)

        loss = self._get_loss()
        penalty = self._get_penalty()

        df = np.zeros((n_samples, n_vectors), dtype=np.float64)
        coef = np.zeros((n_vectors, n_features), dtype=np.float64)
        G = np.zeros((n_vectors, n_features), dtype=np.float64)

        obj = self.C * loss.objective(df, y)
        obj += self.alpha * penalty.regularization(coef)

        L = 1.0
        for t in xrange(self.max_iter):
            if self.verbose >= 1:
                print "Iter", t + 1

            # Save current values
            coef_old = coef
            obj_old = obj

            # Gradient
            G = loss.gradient(df, X, y)
            G *= self.C

            for tt in xrange(self.max_steps):
                # Solve
                coef = coef_old - G / L

                penalty.projection(coef, self.alpha, L)

                s = coef - coef_old
                ss = np.sum(s ** 2)

                df = safe_sparse_dot(X, coef.T)
                obj = self.C * loss.objective(df, y)
                obj += self.alpha * penalty.regularization(coef)

                obj_diff = obj - obj_old
                accepted = obj_diff <= - 0.5 * self.sigma * L * ss

                if accepted:
                    if self.verbose >= 2:
                        print "Accepted at", tt + 1
                        print "obj_diff =", obj_diff
                    break
                else:
                    L *= self.eta
            # end for line search

            L *= self.L_factor
            L = min(self.Lmax, max(self.Lmin, L))

        self.coef_ = coef

        return self

    def decision_function(self, X):
        return safe_sparse_dot(X, self.coef_.T)
