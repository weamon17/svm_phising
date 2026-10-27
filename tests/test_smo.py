import numpy as np
import pytest
from sklearn.datasets import make_classification, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.smo import SMOSVC, kernel_matrix


def test_three_point_hand_example_in_one_step():
    # Worked example of the report: x1=(1,1) y=-1, x2=(2,2) y=+1, x3=(3,3) y=+1.
    X, y = [[1, 1], [2, 2], [3, 3]], [-1, 1, 1]
    model = SMOSVC(C=10, kernel="linear", tol=1e-6, record_trace=True).fit(X, y)

    np.testing.assert_allclose(model.alpha_, [1, 1, 0])
    np.testing.assert_allclose(model.coef_, [1, 1])
    assert model.intercept_ == pytest.approx(-3)
    assert model.support_.tolist() == [0, 1]          # x3 is not a support vector
    np.testing.assert_allclose(model.decision_function(X), [-1, 1, 3])
    assert model.dual_objective_ == pytest.approx(-1)  # max of 2α − α² is 1 at α = 1

    (step,) = model.trace_
    assert (step["i"], step["j"]) == (1, 0)
    assert (step["E_i"], step["E_j"], step["eta"]) == (-1, 1, 2)
    assert (step["L"], step["H"]) == (0, 10)
    assert step["alpha_i"] == (0, 1) and step["alpha_j"] == (0, 1)


def test_kernels_match_sklearn_definitions():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    np.testing.assert_allclose(kernel_matrix(A, B, "linear"), A @ B.T)
    np.testing.assert_allclose(kernel_matrix(A, B, "poly", gamma=0.5, degree=2, coef0=1),
                               (0.5 * A @ B.T + 1) ** 2)
    expected = np.exp(-0.7 * ((A[:, None] - B[None]) ** 2).sum(-1))
    np.testing.assert_allclose(kernel_matrix(A, B, "rbf", gamma=0.7), expected)


def test_solution_satisfies_dual_constraints_and_kkt():
    X, y = make_moons(200, noise=0.25, random_state=1)
    C, tol = 2.0, 1e-4
    model = SMOSVC(C=C, kernel="rbf", gamma=1.0, tol=tol).fit(X, y)
    y_pm = np.where(y == 1, 1.0, -1.0)
    alpha = model.alpha_

    assert model.converged_
    assert np.all((alpha >= 0) & (alpha <= C))
    assert abs(alpha @ y_pm) < 1e-9
    margin = y_pm * model.decision_function(X)
    slack = 10 * tol
    assert np.all(margin[alpha == 0] >= 1 - slack)                       # outside the margin
    assert np.all(np.abs(margin[(alpha > 0) & (alpha < C)] - 1) <= slack)  # on the margin
    assert np.all(margin[alpha == C] <= 1 + slack)                       # inside / misclassified


@pytest.mark.parametrize("params", [
    dict(kernel="linear", C=1.0),
    dict(kernel="rbf", C=1.0, gamma="scale"),
    dict(kernel="rbf", C=10.0, gamma=0.5),
    dict(kernel="poly", C=1.0, gamma=0.5, degree=3, coef0=1.0),
])
def test_matches_sklearn_svc(params):
    X, y = make_classification(400, 6, n_informative=4, random_state=0)
    X = StandardScaler().fit_transform(X)
    X_train, y_train, X_test = X[:300], y[:300], X[300:]

    ours = SMOSVC(tol=1e-5, **params).fit(X_train, y_train)
    ref = SVC(tol=1e-5, **params).fit(X_train, y_train)

    assert ours.converged_
    # sklearn groups support vectors by class; align both by training index before comparing.
    order = np.argsort(ref.support_)
    assert ours.support_.tolist() == ref.support_[order].tolist()
    np.testing.assert_allclose(ours.dual_coef_, ref.dual_coef_[0][order], atol=1e-3)
    assert ours.intercept_ == pytest.approx(ref.intercept_[0], abs=1e-3)
    np.testing.assert_allclose(ours.decision_function(X_test), ref.decision_function(X_test), atol=1e-3)
    np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))


def test_rejects_multiclass():
    with pytest.raises(ValueError):
        SMOSVC().fit([[0], [1], [2]], [0, 1, 2])
