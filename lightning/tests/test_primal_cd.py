import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, \
                       assert_not_equal

from sklearn.datasets.samples_generator import make_classification
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer

from lightning.primal_cd import CDClassifier, CDClassifier
from lightning.primal_cd import C_lower_bound

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
mult_csc = sp.csc_matrix(mult_dense)


def test_fit_linear_binary_l1r():
    clf = CDClassifier(C=1.0, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)
    n_nz = np.sum(clf.coef_ != 0)

    clf = CDClassifier(C=0.1, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 0.97)
    n_nz2 = np.sum(clf.coef_ != 0)

    assert_true(n_nz > n_nz2)


def test_fit_rbf_binary_l1r():
    clf = CDClassifier(C=0.5, kernel="rbf", gamma=0.1, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 0.845)
    n_nz = np.sum(clf.coef_ != 0)
    assert_equal(n_nz, 160)

    K = pairwise_kernels(bin_dense, metric="rbf", gamma=0.1)
    clf2 = CDClassifier(C=0.5, random_state=0, penalty="l1")
    clf2.fit(K, bin_target)
    acc = clf2.score(K, bin_target)
    assert_almost_equal(acc, 0.845)
    n_nz = np.sum(clf2.coef_ != 0)
    assert_equal(n_nz, 160)


def test_fit_rbf_binary_l1r_selection():
    for selection in ("loss", "active"):
        clf = CDClassifier(C=0.5, kernel="rbf", gamma=0.1, random_state=0,
                           penalty="l1", selection=selection)
        clf.fit(bin_dense, bin_target)
        acc = clf.score(bin_dense, bin_target)
        assert_true(acc >= 0.74)
        n_nz = np.sum(clf.coef_ != 0)
        assert_true(n_nz <= 86)


def test_fit_rbf_multi():
    clf = CDClassifier(penalty="l1", kernel="rbf", gamma=0.1, random_state=0)
    clf.fit(mult_dense, mult_target)
    y_pred = clf.predict(mult_dense)
    acc = np.mean(y_pred == mult_target)
    assert_almost_equal(acc, 1.0)


def test_warm_start_l1r():
    clf = CDClassifier(warm_start=True, random_state=0, penalty="l1")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    n_nz2 = np.sum(clf.coef_ != 0)

    assert_true(n_nz < n_nz2)


def test_warm_start_l1r_rbf():
    clf = CDClassifier(warm_start=True, kernel="rbf", gamma=0.1,
                       random_state=0, penalty="l1")

    clf.C = 0.5
    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    clf.C = 0.6
    clf.fit(bin_dense, bin_target)
    n_nz2 = np.sum(clf.coef_ != 0)

    assert_true(n_nz < n_nz2)


def test_early_stopping_l1r_rbf():
    clf = CDClassifier(kernel="rbf", gamma=0.1,
                    termination="n_components", n_components=30,
                    random_state=0, penalty="l1")

    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    assert_equal(n_nz, 30)


def test_fit_linear_binary_l1r_log_loss():
    clf = CDClassifier(C=1.0, random_state=0, penalty="l1", loss="log")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 0.995)


def test_fit_linear_binary_l2r():
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)

    K = pairwise_kernels(bin_dense, metric="rbf", gamma=0.1)
    clf2 = CDClassifier(C=0.5, random_state=0, penalty="l2")
    clf2.fit(K, bin_target)
    acc = clf2.score(K, bin_target)
    assert_almost_equal(acc, 1.0)
    n_nz = np.sum(clf2.coef_ != 0)
    assert_equal(n_nz, 200)


def test_fit_linear_binary_l2r_log():
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2", loss="log",
                       max_iter=5)
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)


def test_fit_rbf_binary_l2r_log():
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2", loss="log",
                       max_iter=5, kernel="rbf")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)


def test_fit_linear_binary_l2r_modified_huber():
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2",
                       loss="modified_huber")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)


def test_fit_rbf_binary_l2r_modified_huber():
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2",
                       kernel="rbf", loss="modified_huber")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)


def test_fit_rbf_binary_l2r():
    clf = CDClassifier(C=0.5, kernel="rbf", gamma=0.1, random_state=0, penalty="l2")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)
    n_nz = np.sum(clf.coef_ != 0)
    assert_equal(n_nz, 200) # dense solution...


def test_fit_linear_multi_l2r():
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2")
    clf.fit(mult_dense, mult_target)
    acc = clf.score(mult_dense, mult_target)
    assert_almost_equal(acc, 0.8833, 4)


def test_fit_rbf_multi_l2r():
    clf = CDClassifier(C=0.5, kernel="rbf", gamma=0.1, random_state=0, penalty="l2")
    clf.fit(mult_dense, mult_target)
    acc = clf.score(mult_dense, mult_target)
    assert_almost_equal(acc, 1.0)


def test_warm_start_l2r():
    clf = CDClassifier(warm_start=True, random_state=0, penalty="l2")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)


def test_warm_start_l2r_rbf():
    clf = CDClassifier(warm_start=True, kernel="rbf", gamma=0.1,
                       random_state=0, penalty="l2")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)


def test_debiasing():
    clf = CDClassifier(kernel="rbf", gamma=0.1, penalty="l1", debiasing=True,
                       C=0.5, Cd=1.0, max_iter=10)
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_nonzero(), 160)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.845)
    pred = clf.decision_function(bin_dense)

    clf = CDClassifier(kernel="rbf", gamma=0.1, penalty="l1", C=0.5, max_iter=10)
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_nonzero(), 160)
    K = pairwise_kernels(bin_dense, clf.support_vectors_, metric="rbf", gamma=0.1)
    clf = CDClassifier(max_iter=10, C=1.0, penalty="l2")
    clf.fit(K, bin_target)
    assert_almost_equal(clf.score(K, bin_target), 0.845)
    pred2 = clf.decision_function(K)

    assert_array_almost_equal(pred, pred2)

def test_debiasing_warm_start():
    clf = CDClassifier(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10)
    clf.C = 0.5
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_nonzero(), 160)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.845)

    clf = CDClassifier(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10)
    clf.C = 0.500001
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_nonzero(), 191)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.97)

    clf = CDClassifier(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10,
                       warm_start=True)
    clf.C = 0.5
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_nonzero(), 160)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.845)

    clf = CDClassifier(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10,
                       warm_start=True)
    clf.C = 0.500001
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_nonzero(), 191)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.97)


def test_early_stopping_l2r_rbf():
    clf = CDClassifier(kernel="rbf", gamma=0.1,
                      termination="n_components", n_components=30,
                      random_state=0, penalty="l2")

    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    assert_equal(n_nz, 30)


def test_empty_model():
    clf = CDClassifier(kernel="rbf", gamma=0.1, C=0.1, penalty="l1")
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_nonzero(), 0)
    acc = clf.score(bin_dense, bin_target)
    assert_equal(acc, 0.5)

    clf = CDClassifier(kernel="rbf", gamma=0.1, C=0.1, penalty="l1",
                       debiasing=True)
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_nonzero(), 0)
    acc = clf.score(bin_dense, bin_target)
    assert_equal(acc, 0.5)


def test_lower_bound_binary():
    Cmin = C_lower_bound(bin_dense, bin_target)
    clf = CDClassifier(C=Cmin, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)
    assert_equal(0, n_nz)

    clf = CDClassifier(C=Cmin * 2, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)
    assert_not_equal(0, n_nz)


def test_lower_bound_multi():
    Cmin = C_lower_bound(mult_dense, mult_target)
    assert_almost_equal(Cmin, 0.00176106681581)


def test_lower_bound_binary_rbf():
    K = pairwise_kernels(bin_dense, metric="rbf", gamma=0.1)
    Cmin = C_lower_bound(K, bin_target)
    Cmin2 = C_lower_bound(bin_dense, bin_target, kernel="rbf", gamma=0.1)
    assert_almost_equal(Cmin, Cmin2, 4)
    Cmin3 = C_lower_bound(bin_dense, bin_target, kernel="rbf", gamma=0.1,
                          search_size=60, random_state=0)
    assert_almost_equal(Cmin, Cmin3, 4)


def test_lower_bound_multi_rbf():
    K = pairwise_kernels(mult_dense, metric="rbf", gamma=0.1)
    Cmin = C_lower_bound(K, mult_target)
    Cmin2 = C_lower_bound(mult_dense, mult_target, kernel="rbf", gamma=0.1)
    Cmin3 = C_lower_bound(mult_dense, mult_target, kernel="rbf", gamma=0.1,
                          search_size=60, random_state=0)
    assert_almost_equal(Cmin, Cmin2, 4)
    assert_almost_equal(Cmin, Cmin3, 4)


def test_components():
    clf = CDClassifier(random_state=0, penalty="l1", kernel="rbf",
                       gamma=0.1, C=0.5)
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)

    clf = CDClassifier(random_state=0, penalty="l2", kernel="rbf",
                       gamma=0.1, C=0.5, components=clf.support_vectors_)
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_nonzero(), 160)
    acc2 = clf.score(bin_dense, bin_target)
    assert_equal(acc, acc2)


def test_fit_rbf_binary_l2r_correctness():
    for loss in ("squared_hinge", "modified_huber", "log"):
        clf = CDClassifier(C=1.0, random_state=0, penalty="l2", loss=loss,
                           max_iter=1, kernel="rbf")
        clf.fit(bin_dense, bin_target)
        acc = clf.score(bin_dense, bin_target)
        assert_almost_equal(acc, 1.0)


def test_fit_squared_loss():
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2",
                       loss="squared", max_iter=100)
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.99)
    y = bin_target.copy()
    y[y == 0] = -1
    assert_array_almost_equal(np.dot(bin_dense, clf.coef_.ravel()) - y,
                              clf.errors_.ravel())

    K = pairwise_kernels(bin_dense, metric="rbf", gamma=0.1)

    clf = CDClassifier(C=1.0, random_state=0, penalty="l2",
                       kernel="rbf", gamma=0.1,
                       loss="squared", max_iter=100)
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)
    assert_array_almost_equal(np.dot(K, clf.coef_.ravel()) - y,
                              clf.errors_.ravel())


def test_fit_squared_loss_l1():
    clf = CDClassifier(C=0.5, random_state=0, penalty="l1",
                       loss="squared", max_iter=100)
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.985, 3)
    y = bin_target.copy()
    y[y == 0] = -1
    assert_array_almost_equal(np.dot(bin_dense, clf.coef_.ravel()) - y,
                              clf.errors_.ravel())
    n_nz = np.sum(clf.coef_ != 0)
    assert_equal(n_nz, 89)

    K = pairwise_kernels(bin_dense, metric="rbf", gamma=0.1)

    clf = CDClassifier(C=0.5, random_state=0, penalty="l1",
                       kernel="rbf", gamma=0.1,
                       loss="squared", max_iter=100)
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.845, 3)
    n_nz = np.sum(clf.coef_ != 0)
    assert_equal(n_nz, 160)


def test_l1l2_multiclass_log_loss():
    for data in (mult_dense, mult_csc):
        clf = CDClassifier(penalty="l1/l2", loss="log", multiclass=True,
                           max_iter=5, C=1.0, random_state=0)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.8733, 3)
        df = clf.decision_function(data)
        sel = np.array([df[i, int(mult_target[i])] for i in xrange(df.shape[0])])
        df -= sel[:, np.newaxis]
        df = np.exp(df)
        assert_array_almost_equal(clf.errors_, df.T)
        assert_equal(np.sum(clf.coef_ != 0), 297)

        clf = CDClassifier(penalty="l1/l2", loss="log", multiclass=True,
                           max_iter=5, C=0.3, random_state=0)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.8366, 3)
        nz = np.sum(clf.coef_ != 0)
        assert_equal(nz, 243)
        assert_true(nz % 3 == 0) # should be a multiple of n_classes


def test_l1l2_multiclass_squared_hinge_loss():
    for data in (mult_dense, mult_csc):
        clf = CDClassifier(penalty="l1/l2", loss="squared_hinge",
                           multiclass=True,
                           max_iter=20, C=1.0, random_state=0)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.9, 3)
        df = clf.decision_function(data)
        n_samples, n_vectors = df.shape
        diff = np.zeros_like(clf.errors_)
        for i in xrange(n_samples):
            for k in xrange(n_vectors):
                diff[k, i] = 1 - (df[i, mult_target[i]] - df[i, k])
        assert_array_almost_equal(clf.errors_, diff)
        assert_equal(np.sum(clf.coef_ != 0), 300)

        clf = CDClassifier(penalty="l1/l2", loss="squared_hinge",
                           multiclass=True,
                           max_iter=20, C=0.05, random_state=0)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.83, 3)
        nz = np.sum(clf.coef_ != 0)
        assert_equal(nz, 219)
        assert_true(nz % 3 == 0) # should be a multiple of n_classes


def test_l1l2_multiclass_squared_hinge_loss_kernel():
    for data in (mult_dense, ):
        clf = CDClassifier(penalty="l1/l2", loss="squared_hinge", multiclass=True,
                           kernel="rbf", gamma=0.1,
                           max_iter=20, C=1.0, random_state=0)
        clf.fit(data, mult_target)
        assert_equal(clf.score(data, mult_target), 1.0)
        assert_equal(clf.n_nonzero(), 300)


def test_l1l2_multi_task_squared_hinge_loss():
    Y = LabelBinarizer(neg_label=-1).fit_transform(mult_target)
    clf = CDClassifier(penalty="l1/l2", loss="squared_hinge",
                       multiclass=False,
                       max_iter=20, C=5.0, random_state=0)
    clf.fit(mult_dense, mult_target)
    df = clf.decision_function(mult_dense)
    assert_array_almost_equal(clf.errors_.T, 1 - Y * df)
    assert_almost_equal(clf.score(mult_dense, mult_target), 0.8633, 3)
    nz = np.sum(clf.coef_ != 0)
    assert_equal(nz, 300)

    clf = CDClassifier(penalty="l1/l2", loss="squared_hinge",
                       multiclass=False,
                       max_iter=20, C=0.05, random_state=0)
    clf.fit(mult_dense, mult_target)
    assert_almost_equal(clf.score(mult_dense, mult_target), 0.8266, 3)
    nz = np.sum(clf.coef_ != 0)
    assert_equal(nz, 240)


def test_l1l2_multi_task_log_loss():
    clf = CDClassifier(penalty="l1/l2", loss="log",
                       multiclass=False,
                       max_iter=20, C=5.0, random_state=0)
    clf.fit(mult_dense, mult_target)
    assert_almost_equal(clf.score(mult_dense, mult_target), 0.8633, 3)


def test_l1l2_multi_task_square_loss():
    clf = CDClassifier(penalty="l1/l2", loss="squared",
                       multiclass=False,
                       max_iter=20, C=5.0, random_state=0)
    clf.fit(mult_dense, mult_target)
    assert_almost_equal(clf.score(mult_dense, mult_target), 0.8066, 3)

