import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, \
                       assert_not_equal

from sklearn.datasets.samples_generator import make_classification
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state

from lightning.dataset_fast import ContiguousDataset
from lightning.dataset_fast import FortranDataset
from lightning.dataset_fast import CSRDataset
from lightning.dataset_fast import CSCDataset
from lightning.dataset_fast import KernelDataset

X, _ = make_classification(n_samples=20, n_features=100,
                           n_informative=5, n_classes=2, random_state=0)
X2, _ = make_classification(n_samples=10, n_features=100,
                            n_informative=5, n_classes=2, random_state=0)

X_csr = sp.csr_matrix(X)
X_csc = sp.csc_matrix(X)

rs = check_random_state(0)


def test_contiguous_dot():
    ds = ContiguousDataset(X)
    assert_array_almost_equal(ds.dot(X2.T), np.dot(X, X2.T))


def test_fortran_dot():
    ds = FortranDataset(np.asfortranarray(X))
    assert_array_almost_equal(ds.dot(X2.T), np.dot(X, X2.T))


def test_csr_dot():
    ds = CSRDataset(X_csr)
    assert_array_almost_equal(ds.dot(X2.T), np.dot(X, X2.T))


def test_csc_dot():
    ds = CSCDataset(X_csc)
    assert_array_almost_equal(ds.dot(X2.T), np.dot(X, X2.T))


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


def test_kernel_dot():
    coef = rs.randn(X2.shape[0], 3)
    K = pairwise_kernels(X, X2, metric="rbf", gamma=0.1)
    kd = KernelDataset(X, X2, "rbf", gamma=0.1)
    assert_array_almost_equal(kd.dot(coef),
                              np.dot(K, coef))
