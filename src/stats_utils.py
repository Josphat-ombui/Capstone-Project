"""Small statistical helpers: Wilson intervals, exact Poisson intervals and a logistic regression
with standard errors (written with numpy/scipy so no extra package is needed)."""
import numpy as np
import pandas as pd
from scipy import stats


def wilson_interval(failures, n, z=1.959964):
    """95% Wilson score interval for a proportion."""
    failures, n = np.asarray(failures, float), np.asarray(n, float)
    p = failures / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


def poisson_ratio_interval(observed, expected, level=0.95):
    """Exact (Garwood) interval for observed / expected counts."""
    observed, expected = np.asarray(observed, float), np.asarray(expected, float)
    a = (1 - level) / 2
    low = np.where(observed > 0, stats.chi2.ppf(a, 2 * observed) / 2, 0.0)
    high = stats.chi2.ppf(1 - a, 2 * observed + 2) / 2
    return low / expected, high / expected


def _independent_columns(X, tol=1e-8):
    """Keep columns in order, dropping any that are a linear combination of earlier ones."""
    keep = []
    for j in range(X.shape[1]):
        trial = X[:, keep + [j]]
        if np.linalg.matrix_rank(trial, tol=tol * max(1.0, np.abs(trial).max())) == len(keep) + 1:
            keep.append(j)
    return keep


class LogitResult:
    """Coefficients, standard errors and p-values of a fitted logistic regression."""

    def __init__(self, params, bse, columns):
        self.params = pd.Series(params, index=columns)
        self.bse = pd.Series(bse, index=columns)
        z = self.params / self.bse
        self.pvalues = pd.Series(2 * stats.norm.sf(np.abs(z)), index=columns)

    def predict(self, X):
        X = X.reindex(columns=self.params.index, fill_value=0.0)
        return 1 / (1 + np.exp(-(X.to_numpy(float) @ self.params.to_numpy())))


def fit_logit(X, y, max_iter=100, tol=1e-10):
    """Maximum-likelihood logistic regression by Newton-Raphson. X must include a 'const' column."""
    X = X.astype(float)
    cols = list(X.columns)
    keep = _independent_columns(X.to_numpy())
    cols = [cols[j] for j in keep]
    A = X[cols].to_numpy()
    yv = np.asarray(y, float)
    beta = np.zeros(A.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-(A @ beta)))
        W = p * (1 - p)
        grad = A.T @ (yv - p)
        H = A.T @ (A * W[:, None])
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    p = 1 / (1 + np.exp(-(A @ beta)))
    H = A.T @ (A * (p * (1 - p))[:, None])
    bse = np.sqrt(np.diag(np.linalg.inv(H)))
    return LogitResult(beta, bse, cols)
