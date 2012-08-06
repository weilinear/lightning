# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state

from .base import BaseClassifier

from .dataset_fast import KernelDataset
from .primal_cd_fast import _primal_cd
from .primal_cd_fast import _C_lower_bound_kernel

from .primal_cd_fast import Squared
from .primal_cd_fast import SquaredHinge
from .primal_cd_fast import ModifiedHuber
from .primal_cd_fast import Log


class BaseCD(object):

    def _get_loss(self):
        params = {"max_steps" : self.max_steps,
                  "sigma" : self.sigma,
                  "beta" : self.beta,
                  "verbose" : self.verbose}
        losses = {
            "squared" : Squared(verbose=self.verbose),
            "squared_hinge" : SquaredHinge(**params),
            "modified_huber" : ModifiedHuber(**params),
            "log" : Log(**params),
        }
        return losses[self.loss]


class CDClassifier(BaseCD, BaseClassifier, ClassifierMixin):

    def __init__(self, C=1.0, loss="squared_hinge", penalty="l2",
                 multiclass=False,
                 max_iter=50, tol=1e-3, termination="convergence",
                 shrinking=True,
                 max_steps=30, sigma=0.01, beta=0.5,
                 kernel=None, gamma=0.1, coef0=1, degree=4, cache_mb=500,
                 warm_start=False, debiasing=False, Cd=1.0,
                 warm_debiasing=False,
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
        self.shrinking = shrinking
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
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
        self.violation_init_ = {}

    def fit(self, X, y):
        rs = self._get_random_state()

        # Create dataset
        X, A, ds = self._get_dataset(X, self.components, order="fortran")
        self.support_vectors_ = A
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()

        # Create label transformers
        reencode = self.penalty == "l1/l2"
        y, n_classes, n_vectors = self._set_label_transformers(y, reencode)
        Y = np.asfortranarray(self.label_binarizer_.transform(y),
                              dtype=np.float64)

        # Initialize coefficients
        if self.warm_start and self.coef_ is not None:
            if self.kernel:
                coef = np.zeros((n_vectors, n_features), dtype=np.float64)
                coef[:, self.support_indices_] = self.coef_
                self.coef_ = coef
        else:
            self.C_init = self.C
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)

            if self.loss == "squared":
                self.errors_ -= 1 + Y.T

        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)
        indices = np.arange(n_features, dtype=np.int32)

        # Learning
        if self.penalty == "l1/l2":
            vinit = self.violation_init_.get(0, 0) * self.C / self.C_init
            viol = _primal_cd(self, self.coef_, self.errors_,
                              ds, y, Y, -1, self.multiclass,
                              indices, 12, self._get_loss(),
                              self.selection, self.search_size,
                              self.termination, self.n_components,
                              self.C, self.max_iter, self.shrinking, vinit,
                              rs, self.tol, self.callback, self.verbose)
            if self.warm_start and len(self.violation_init_) == 0:
                self.violation_init_[0] = viol

        elif self.penalty in ("l1", "l2"):
            penalty = 1 if self.penalty == "l1" else 2
            for k in xrange(n_vectors):
                vinit = self.violation_init_.get(k, 0) * self.C / self.C_init
                viol = _primal_cd(self, self.coef_, self.errors_,
                                  ds, y, Y, k, False,
                                  indices, penalty, self._get_loss(),
                                  self.selection, self.search_size,
                                  self.termination, self.n_components,
                                  self.C, self.max_iter, self.shrinking, vinit,
                                  rs, self.tol, self.callback, self.verbose)
                if self.warm_start and not k in self.violation_init_:
                    self.violation_init_[k] = viol

        if self.debiasing:
            nz = np.sum(self.coef_ != 0, axis=0, dtype=bool)
            self.support_indices_ = np.arange(n_features, dtype=np.int32)[nz]
            indices = self.support_indices_.copy()
            if not self.warm_debiasing:
                self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
                self.errors_ = np.ones((n_vectors, n_features), dtype=np.float64)

            for k in xrange(n_vectors):
                _primal_cd(self, self.coef_, self.errors_,
                           ds, y, Y, k, False,
                           indices, 2, self._get_loss(),
                           "permute", self.search_size,
                           "convergence", self.n_components,
                           self.Cd, self.max_iter, self.shrinking, 0,
                           rs, self.tol, self.callback, self.verbose)

        nz = np.sum(self.coef_ != 0, axis=0, dtype=bool)
        self.support_indices_ = np.arange(n_features, dtype=np.int32)[nz]

        if self.kernel:
            self._post_process(A)

        return self

    def decision_function(self, X):
        X, _, ds = self._get_dataset(X, self.support_vectors_)
        return ds.dot(self.coef_.T) + self.intercept_


def C_lower_bound(X, y, kernel=None, search_size=None, random_state=None,
                  **kernel_params):
    Y = LabelBinarizer(neg_label=-1, pos_label=1).fit_transform(y)
    Y = np.asfortranarray(Y, dtype=np.float64)

    if kernel is None:
        den = np.max(np.abs(np.dot(Y.T, X)))
    else:
        random_state = check_random_state(random_state)
        kds = KernelDataset(X, X, kernel=kernel, **kernel_params)
        den = _C_lower_bound_kernel(kds, Y, search_size, random_state)

    if den == 0.0:
        raise ValueError('Ill-posed')

    return 0.5 / den
