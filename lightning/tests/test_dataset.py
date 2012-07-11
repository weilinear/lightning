import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, \
                       assert_not_equal

from sklearn.datasets.samples_generator import make_classification
from sklearn.metrics.pairwise import pairwise_kernels

from lightning.dataset_fast import KernelDataset

X, _ = make_classification(n_samples=20, n_features=100,
                           n_informative=5, n_classes=2, random_state=0)


def check_kernel(K, kd):
    for i in xrange(K.shape[0]):
        indices, data, n_nz = kd.get_column(i)
        assert_array_almost_equal(K[i], data)
        assert_equal(n_nz, K.shape[0])


def test_dataset_linear_kernel():
    K = pairwise_kernels(X, metric="linear")
    kd = KernelDataset(X, X, "linear")
    check_kernel(K, kd)


def test_dataset_poly_kernel():
    K = pairwise_kernels(X, metric="poly", gamma=0.1, coef0=1, degree=4)
    kd = KernelDataset(X, X, "poly", gamma=0.1, coef0=1, degree=4)
    check_kernel(K, kd)


def test_dataset_rbf_kernel():
    K = pairwise_kernels(X, metric="rbf", gamma=0.1)
    kd = KernelDataset(X, X, "rbf", gamma=0.1)
    check_kernel(K, kd)
