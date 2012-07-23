# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import ClassifierMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot

from .base import BaseClassifier

from .sgd_fast import _binary_sgd
from .sgd_fast import _multiclass_hinge_sgd
from .sgd_fast import _multiclass_log_sgd

from .sgd_fast import ModifiedHuber
from .sgd_fast import Hinge
from .sgd_fast import Log
from .sgd_fast import SparseLog
from .sgd_fast import SquaredLoss
from .sgd_fast import Huber
from .sgd_fast import EpsilonInsensitive


class SGDClassifier(BaseClassifier, ClassifierMixin):

    def __init__(self, loss="hinge", multiclass="one-vs-rest", lmbda=0.01,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 learning_rate="pegasos", eta0=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 n_components=0, max_iter=10, random_state=None,
                 cache_mb=500, verbose=0, n_jobs=1):
        self.loss = loss
        self.multiclass = multiclass
        self.lmbda = lmbda
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_decay = intercept_decay
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.cache_mb = cache_mb
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None

    def _get_loss(self):
        losses = {
            "modified_huber" : ModifiedHuber(),
            "hinge" : Hinge(1.0),
            "perceptron" : Hinge(0.0),
            "log": Log(),
            "sparse_log" : SparseLog(),
            "squared" : SquaredLoss(),
            "huber" : Huber(self.epsilon),
            "epsilon_insensitive" : EpsilonInsensitive(self.epsilon)
        }
        return losses[self.loss]

    def _get_learning_rate(self):
        learning_rates = {"constant": 1, "pegasos": 2, "invscaling": 3}
        return learning_rates[self.learning_rate]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rs = check_random_state(self.random_state)

        reencode = self.multiclass == "natural"
        y, n_classes, n_vectors = self._set_label_transformers(y, reencode)

        if self.kernel == "linear":
            ds = self._get_dataset(X, kernel=False)
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        else:
            ds = self._get_dataset(X)
            self.coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)

        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        if n_vectors == 1 or self.multiclass == "one-vs-rest":
            Y = self.label_binarizer_.transform(y)
            for i in xrange(n_vectors):
                _binary_sgd(self,
                            self.coef_, self.intercept_, i,
                            ds, Y[:, i],
                            self._get_loss(),
                            self.n_components,
                            self.lmbda,
                            self._get_learning_rate(),
                            self.eta0, self.power_t,
                            self.fit_intercept,
                            self.intercept_decay,
                            self.max_iter * n_samples,
                            rs, self.verbose)

        elif self.multiclass == "natural":
            if self.loss in ("hinge", "log"):
                func = eval("_multiclass_%s_sgd" % self.loss)
                func(self, self.coef_, self.intercept_,
                     ds, y.astype(np.int32),
                     self.n_components,
                     self.lmbda, self._get_learning_rate(), self.eta0,
                     self.power_t, self.fit_intercept, self.intercept_decay,
                     self.max_iter * n_samples, rs, self.verbose)
            else:
                raise ValueError("Loss not supported for multiclass!")

        else:
            raise ValueError("Wrong value for multiclass.")

        if self.kernel != "linear":
            self._post_process(X)

        return self

    def decision_function(self, X):
        if self.kernel == "linear":
            return safe_sparse_dot(X, self.coef_.T) + self.intercept_
        else:
            ds = self._get_dataset(X, self.support_vectors_)
            return ds.dot(self.coef_.T) + self.intercept_
