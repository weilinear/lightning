# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

from sklearn.base import ClassifierMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.utils import safe_asarray
from sklearn.utils import safe_mask

from .base import BaseClassifier

from .kernel_fast import get_kernel, KernelCache
from .primal_cd_fast import _primal_cd_l1l2r
from .primal_cd_fast import _primal_cd_l2r
from .primal_cd_fast import _primal_cd_l1r
from .primal_cd_fast import _C_lower_bound_kernel

from .primal_cd_fast import Squared
from .primal_cd_fast import SquaredHinge
from .primal_cd_fast import ModifiedHuber
from .primal_cd_fast import Log


class BaseCD(object):

    def _get_loss(self):
        losses = {
            "squared" : Squared(),
            "squared_hinge" : SquaredHinge(),
            "modified_huber" : ModifiedHuber(),
            "log" : Log(),
        }
        return losses[self.loss]


class CDClassifier(BaseCD, BaseClassifier, ClassifierMixin):

    def __init__(self, C=1.0, loss="squared_hinge", penalty="l2",
                 multiclass=False,
                 max_iter=50, tol=1e-3, termination="convergence",
                 kernel=None, gamma=0.1, coef0=1, degree=4, cache_mb=500,
                 warm_start=False, debiasing=False, Cd=1.0, warm_debiasing=False,
                 selection="permute", search_size=60,
                 n_components=1000, components=None,
                 random_state=None, callback=None, verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.penalty = penalty
        self.multiclass = multiclass
        self.max_iter = max_iter
        self.tol = tol
        self.termination = termination
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.cache_mb = cache_mb
        self.warm_start = warm_start
        self.debiasing = debiasing
        self.Cd = Cd
        self.warm_debiasing = warm_debiasing
        self.selection = selection
        self.search_size = search_size
        self.n_components = n_components
        self.components = components
        self.random_state = random_state
        self.callback = callback
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.support_vectors_ = None
        self.coef_ = None

    def fit(self, X, y, kcache=None):
        rs = self._get_random_state()

        # Check data
        if self.kernel:
            X = np.ascontiguousarray(X, dtype=np.float64)
            if self.components is not None:
                A = np.ascontiguousarray(self.components, dtype=np.float64)
            else:
                A = X
            self.support_vectors_ = A
        else:
            if sp.issparse(X):
                X = X.tocsc()
            else:
                X = np.asfortranarray(X, dtype=np.float64)
            A = None

        # Create dataset
        ds = self._get_dataset(X, A)
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()

        # Create label transformers
        reencode = self.penalty == "l1/l2"
        y, n_classes, n_vectors = self._set_label_transformers(y, reencode)
        Y = np.asfortranarray(self.label_binarizer_.transform(y))

        # Initialize coefficients
        if self.warm_start and self.coef_ is not None:
            if self.kernel:
                coef = np.zeros((n_vectors, n_features), dtype=np.float64)
                coef[:, self.support_indices_] = self.coef_
                self.coef_ = coef
        else:
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)

            if self.loss == "squared":
                self.errors_ -= 1 + Y.T

        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)
        indices = np.arange(n_features, dtype=np.int32)

        # Learning
        if self.penalty == "l1/l2":
            _primal_cd_l1l2r(self,
                             self.coef_, self.errors_,
                             ds, y, Y, self.multiclass,
                             indices, self._get_loss(),
                             self.C, self.max_iter, rs, self.tol,
                             self.callback, self.verbose)

        if self.penalty == "l1":
            for i in xrange(n_vectors):
                    _primal_cd_l1r(self, self.coef_[i], self.errors_[i],
                                   ds, Y[:, i],
                                   indices, self._get_loss(),
                                   self.selection, self.search_size,
                                   self.termination, self.n_components,
                                   self.C, self.max_iter, rs, self.tol,
                                   self.callback, verbose=self.verbose)

        if self.penalty == "l2":
            for i in xrange(n_vectors):
                _primal_cd_l2r(self, self.coef_[i], self.errors_[i],
                               ds, Y[:, i],
                               indices, self._get_loss(),
                               self.selection, self.search_size,
                               self.termination, self.n_components,
                               self.C, self.max_iter, rs, self.tol,
                               self.callback, verbose=self.verbose)

        if self.debiasing:
            nz = np.sum(self.coef_ != 0, axis=0, dtype=bool)
            self.support_indices_ = np.arange(n_features, dtype=np.int32)[nz]
            indices = self.support_indices_.copy()
            if not self.warm_debiasing:
                self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
                self.errors_ = np.ones((n_vectors, n_features), dtype=np.float64)

            for i in xrange(n_vectors):
                _primal_cd_l2r(self, self.coef_[i], self.errors_[i],
                               ds, Y[:, i],
                               indices, self._get_loss(),
                               "permute", self.search_size,
                               "convergence", self.n_components,
                               self.Cd, self.max_iter, rs, self.tol,
                               self.callback, verbose=self.verbose)

        nz = np.sum(self.coef_ != 0, axis=0, dtype=bool)
        self.support_indices_ = np.arange(n_features, dtype=np.int32)[nz]

        if np.sum(nz) == 0:
            # Empty model...
            return self

        if self.kernel:
            self._post_process(A)

        return self

    def decision_function(self, X):
        ds = self._get_dataset(X, self.support_vectors_)
        return ds.dot(self.coef_.T) + self.intercept_


def C_lower_bound(X, y, kernel=None, search_size=None, random_state=None,
                  **kernel_params):
    Y = LabelBinarizer(neg_label=-1, pos_label=1).fit_transform(y)

    if kernel is None:
        den = np.max(np.abs(np.dot(Y.T, X)))
    else:
        kernel = get_kernel(kernel, **kernel_params)
        random_state = check_random_state(random_state)
        den = _C_lower_bound_kernel(X, Y, kernel, search_size, random_state)

    if den == 0.0:
        raise ValueError('Ill-posed')

    return 0.5 / den


def C_upper_bound(X, y, clf, Cmin, Cmax, n_components, epsilon, verbose=0):
    Nmax = np.inf
    clf = clone(clf)

    while Nmax - n_components > epsilon:
        Cmid = (Cmin + Cmax) / 2

        if verbose:
            print "Fit clf for C=", Cmid

        clf.set_params(C=Cmid)
        clf.fit(X, y)
        n_nz = clf.n_nonzero()

        if verbose:
            print "#NZ", n_nz

        if n_nz < n_components:
            # Regularization is too strong
            Cmin = Cmid

        elif n_nz > n_components:
            # Regularization is too light
            Cmax = Cmid
            Nmax = n_nz

    if verbose:
        print "Solution: Cmax=", Cmax, "Nmax=", Nmax

    return Cmax
