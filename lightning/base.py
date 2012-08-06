# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.utils import safe_mask
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

from .random import RandomState

from .dataset_fast import ContiguousDataset
from .dataset_fast import FortranDataset
from .dataset_fast import CSRDataset
from .dataset_fast import CSCDataset
from .dataset_fast import KernelDataset


class BaseClassifier(BaseEstimator):

    def predict_proba(self, X):
        if len(self.classes_) != 2:
            raise NotImplementedError("predict_(log_)proba only supported"
                                      " for binary classification")

        if self.loss == "log":
            df = self.decision_function(X).ravel()
            prob = 1.0 / (1.0 + np.exp(-df))
        elif self.loss == "modified_huber":
            df = self.decision_function(X).ravel()
            prob = np.minimum(1, np.maximum(-1, df))
            prob += 1
            prob /= 2
        else:
            raise NotImplementedError("predict_(log_)proba only supported when"
                                      " loss='log' or loss='modified_huber' "
                                      "(%s given)" % self.loss)

        out = np.zeros((X.shape[0], 2), dtype=np.float64)
        out[:, 1] = prob
        out[:, 0] = 1 - prob

        return out

    def _get_random_state(self):
        return RandomState(seed=self.random_state)

    def _set_label_transformers(self, y, reencode=False):
        if reencode:
            self.label_encoder_ = LabelEncoder()
            y = self.label_encoder_.fit_transform(y).astype(np.int32)
        else:
            y = y.astype(np.int32)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        self.label_binarizer_.fit(y)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_classes = len(self.label_binarizer_.classes_)
        n_vectors = 1 if n_classes <= 2 else n_classes

        return y, n_classes, n_vectors

    def n_nonzero(self):
        if hasattr(self, "support_indices_") and \
           self.support_indices_.shape[0] == 0:
            return 0
        elif hasattr(self, "dual_coef_"):
            return np.sum(np.sum(self.dual_coef_ != 0, axis=0, dtype=bool))
        else:
            return np.sum(np.sum(self.coef_ != 0, axis=0, dtype=bool))

    def _post_process(self, X):
        # We can't know the support vectors when using precomputed kernels.
        if self.kernel != "precomputed":
            sv = np.sum(self.coef_ != 0, axis=0, dtype=bool)
            if np.sum(sv) > 0:
                self.coef_ = np.ascontiguousarray(self.coef_[:, sv])
                mask = safe_mask(X, sv)
                self.support_vectors_ = np.ascontiguousarray(X[mask])
                self.support_indices_ = np.arange(X.shape[0], dtype=np.int32)[sv]

            if self.verbose >= 1:
                print "Number of support vectors:", np.sum(sv)

    def _post_process_dual(self, X):
        # We can't know the support vectors when using precomputed kernels.
        if self.kernel != "precomputed":
            sv = np.sum(self.dual_coef_ != 0, axis=0, dtype=bool)
            if np.sum(sv) > 0:
                self.dual_coef_ = np.ascontiguousarray(self.dual_coef_[:, sv])
                mask = safe_mask(X, sv)
                self.support_vectors_ = np.ascontiguousarray(X[mask])
                self.support_indices_ = np.arange(X.shape[0], dtype=np.int32)[sv]

            if self.verbose >= 1:
                print "Number of support vectors:", np.sum(sv)

    def predict(self, X):
        pred = self.decision_function(X)
        out = self.label_binarizer_.inverse_transform(pred, threshold=0)

        if hasattr(self, "label_encoder_"):
            out = self.label_encoder_.inverse_transform(out)

        return out

    def _get_dataset(self, X, Y=None, kernel=True, order="c"):
        if kernel and self.kernel is not None:
            X = np.ascontiguousarray(X, dtype=np.float64)

            if Y is None:
                Y = X
            else:
                Y = np.ascontiguousarray(Y, dtype=np.float64)

            ds = KernelDataset(X, Y, self.kernel,
                               self.gamma, self.coef0, self.degree,
                               self.cache_mb, 1, self.verbose)
        else:
            if sp.isspmatrix(X):
                if order == "fortran":
                    X = X.tocsc()
                    ds = CSCDataset(X)
                else:
                    X = X.tocsr()
                    ds = CSRDataset(X)
            else:
                if order == "fortran":
                    X = np.asfortranarray(X, dtype=np.float64)
                    ds = FortranDataset(X)
                else:
                    X = np.ascontiguousarray(X, dtype=np.float64)
                    ds = ContiguousDataset(X)

        return ds


