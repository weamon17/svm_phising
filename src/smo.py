"""Soft-margin SVM trained by Sequential Minimal Optimization, written from scratch.

Notation (matches the report):
    Dual, minimisation form:  min_α  ½ Σ_i Σ_j α_i α_j y_i y_j K_ij − Σ_i α_i
                              s.t.   0 ≤ α_i ≤ C,   Σ_i α_i y_i = 0
    u_t = Σ_j α_j y_j K_tj     decision value without the bias b
    E_t = u_t − y_t            error; b cancels in every difference E_i − E_j

Working-set selection (which pair to optimise):
    I_up  = {t : y_t = +1, α_t < C} ∪ {t : y_t = −1, α_t > 0}   (y_t α_t can still grow)
    I_low = {t : y_t = +1, α_t > 0} ∪ {t : y_t = −1, α_t < C}   (y_t α_t can still shrink)
    i = argmax_{t ∈ I_up} (−E_t),   m = −E_i,   M = min_{t ∈ I_low} (−E_t)
    m − M is the largest KKT violation; stop when m − M < tol.
    j maximises the second-order gain (m + E_t)² / η_it over t ∈ I_low with −E_t < m
    (Fan, Chen & Lin, 2005: the rule LIBSVM and sklearn.svm.SVC use).

Two-variable sub-problem, solved analytically (Platt, 1998):
    η   = K_ii + K_jj − 2 K_ij
    α_j ← clip(α_j + y_j (E_i − E_j) / η, L, H)
    α_i ← α_i + y_i y_j (α_j_old − α_j)
    E   ← E + Δα_i y_i K[:, i] + Δα_j y_j K[:, j]
"""

import warnings

import numpy as np


TAU = 1e-12  # replaces η ≤ 0 (duplicate points or non-PSD kernels), as in LIBSVM


def kernel_matrix(A, B, kernel="rbf", gamma=1.0, degree=3, coef0=0.0):
    """K[a, b] = k(A[a], B[b]) with the same definitions as sklearn.svm.SVC."""
    if kernel == "linear":
        return A @ B.T
    if kernel == "poly":
        return (gamma * (A @ B.T) + coef0) ** degree
    if kernel == "rbf":
        sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(sq, 0.0))
    raise ValueError(f"unknown kernel {kernel!r}")


class SMOSVC:
    """Binary SVM classifier; fit/predict/decision_function mirror sklearn.svm.SVC."""

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, coef0=0.0,
                 tol=1e-3, max_iter=1_000_000, record_trace=False):
        self.C, self.kernel, self.gamma, self.degree, self.coef0 = C, kernel, gamma, degree, coef0
        self.tol, self.max_iter, self.record_trace = tol, max_iter, record_trace

    def _resolve_gamma(self, X):
        if self.gamma == "scale":
            variance = X.var()
            return 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _kernel(self, A, B):
        return kernel_matrix(A, B, self.kernel, self.gamma_, self.degree, self.coef0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SMOSVC is a binary classifier")
        # Like sklearn: the larger class label is the positive side (decision > 0).
        y_pm = np.where(y == self.classes_[1], 1.0, -1.0)
        self.gamma_ = self._resolve_gamma(X)
        K = self._kernel(X, X)
        diag = np.diag(K).copy()
        n, C = len(X), float(self.C)

        alpha = np.zeros(n)
        E = -y_pm.copy()            # all α = 0 → u = 0 → E = −y
        pos, neg = y_pm > 0, y_pm < 0
        self.trace_ = []
        self.converged_ = False

        for iteration in range(self.max_iter):
            up = (pos & (alpha < C)) | (neg & (alpha > 0))
            low = (pos & (alpha > 0)) | (neg & (alpha < C))
            score = -E
            up_idx = np.flatnonzero(up)
            i = up_idx[np.argmax(score[up_idx])]
            m, M = score[i], score[low].min()
            if m - M < self.tol:
                self.converged_ = True
                break

            eta_all = diag[i] + diag - 2.0 * K[i]
            eta_all = np.where(eta_all > 0, eta_all, TAU)
            gain = np.where(low & (score < m), (m - score) ** 2 / eta_all, -np.inf)
            j = int(np.argmax(gain))

            yi, yj, ai, aj = y_pm[i], y_pm[j], alpha[i], alpha[j]
            eta = eta_all[j]
            if yi != yj:
                L, H = max(0.0, aj - ai), min(C, C + aj - ai)
            else:
                L, H = max(0.0, ai + aj - C), min(C, ai + aj)
            aj_new = min(max(aj + yj * (E[i] - E[j]) / eta, L), H)
            ai_new = ai + yi * yj * (aj - aj_new)
            # Snap round-off to the box so "at bound" tests stay exact.
            ai_new = 0.0 if ai_new < 1e-12 * C else C if ai_new > C * (1 - 1e-12) else ai_new
            aj_new = 0.0 if aj_new < 1e-12 * C else C if aj_new > C * (1 - 1e-12) else aj_new

            if self.record_trace:
                self.trace_.append({
                    "iter": iteration + 1, "i": int(i), "j": j, "kkt_gap": m - M,
                    "E_i": E[i], "E_j": E[j], "eta": eta, "L": L, "H": H,
                    "alpha_i": (ai, ai_new), "alpha_j": (aj, aj_new),
                })
            E += (ai_new - ai) * yi * K[:, i] + (aj_new - aj) * yj * K[:, j]
            alpha[i], alpha[j] = ai_new, aj_new
        else:
            warnings.warn(f"SMO stopped at max_iter={self.max_iter} before reaching tol={self.tol}")

        self.n_iter_ = iteration + 1 if not self.converged_ else iteration
        self.intercept_ = self._bias(alpha, y_pm, E, C)
        u = E + y_pm
        self.dual_objective_ = 0.5 * np.dot(alpha * y_pm, u) - alpha.sum()

        sv = alpha > 0
        self.alpha_ = alpha
        self.support_ = np.flatnonzero(sv)
        self.support_vectors_ = X[sv]
        self.dual_coef_ = (alpha * y_pm)[sv]          # α_i y_i, same as sklearn's dual_coef_
        self.n_support_ = np.array([np.sum(sv & neg), np.sum(sv & pos)])
        return self

    @staticmethod
    def _bias(alpha, y, E, C):
        """b from KKT: E_t = −b on free support vectors (0 < α_t < C); LIBSVM's rule otherwise."""
        free = (alpha > 0) & (alpha < C)
        if free.any():
            return -E[free].mean()
        upper, lower = alpha >= C, alpha <= 0
        ub = E[(upper & (y < 0)) | (lower & (y > 0))].min(initial=np.inf)
        lb = E[(upper & (y > 0)) | (lower & (y < 0))].max(initial=-np.inf)
        return -(ub + lb) / 2

    @property
    def coef_(self):
        if self.kernel != "linear":
            raise AttributeError("coef_ only exists for the linear kernel")
        return self.dual_coef_ @ self.support_vectors_

    def decision_function(self, X):
        """f(x) = Σ_{i ∈ SV} α_i y_i K(x_i, x) + b; its sign is the class, |f| the confidence."""
        X = np.asarray(X, dtype=np.float64)
        return self._kernel(X, self.support_vectors_) @ self.dual_coef_ + self.intercept_

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, self.classes_[1], self.classes_[0])
