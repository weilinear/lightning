import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, \
                       assert_not_equal

from sklearn.datasets.samples_generator import make_classification

from lightning.sparsa import SparsaClassifier
from lightning.primal_cd import CDClassifier

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
mult_csr = sp.csr_matrix(mult_dense)


def test_sparsa_multiclass_l1l2():
    for data in (mult_dense, mult_csr):
        clf = SparsaClassifier(max_iter=500, penalty="l1/l2")
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.97)

def test_sparsa_multiclass_l1():
    for data in (mult_dense, mult_csr):
        clf = SparsaClassifier(max_iter=500, penalty="l1")
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.96)

