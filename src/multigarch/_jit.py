"""JIT-compiled inner loops for GARCH models."""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def garch_variance_loop(
    returns: np.ndarray,
    omega: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    var0: float,
) -> np.ndarray:
    """Compute GARCH(p,q) conditional variance series.

    Each step is clamped to [1e-10, 1e12] so the recursion (and any
    likelihood built on it) stays finite even at infeasible parameters
    probed by the optimizer during line search.

    Args:
        returns: 1D array of returns
        omega: intercept
        alpha: ARCH coefficients (length p)
        beta: GARCH coefficients (length q)
        var0: initial variance estimate

    Returns:
        Conditional variance series
    """
    T = len(returns)
    p = len(alpha)
    q = len(beta)
    max_lag = max(p, q)

    sigma2 = np.full(T, var0)

    for t in range(max_lag, T):
        s = omega
        for i in range(p):
            s += alpha[i] * returns[t - 1 - i] ** 2
        for j in range(q):
            s += beta[j] * sigma2[t - 1 - j]
        if s < 1e-10:
            s = 1e-10
        elif s > 1e12:
            s = 1e12
        sigma2[t] = s

    return sigma2


@njit(cache=True, nogil=True)
def garch_loglik(
    returns: np.ndarray,
    omega: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    var0: float,
) -> float:
    """GARCH(p,q) negative log-likelihood: 0.5 * sum(log s2 + r^2/s2).

    Finite and smooth everywhere; positivity/stationarity are enforced by
    the optimizer's bounds and constraints, not by penalty cliffs.

    Args:
        returns: 1D array of returns
        omega: intercept
        alpha: ARCH coefficients
        beta: GARCH coefficients
        var0: initial variance estimate

    Returns:
        Negative log-likelihood (for minimization)
    """
    if omega < 1e-12:
        omega = 1e-12

    sigma2 = garch_variance_loop(returns, omega, alpha, beta, var0)

    ll = 0.0
    for t in range(len(returns)):
        ll += np.log(sigma2[t]) + returns[t] ** 2 / sigma2[t]

    return 0.5 * ll


@njit(cache=True, nogil=True)
def dcc_loglik_loop(
    std_resid: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> float:
    """DCC negative log-likelihood: 0.5 * sum_{t>=1}(log|R_t| + e' R_t^-1 e).

    Allocation-free inner loop. The Cholesky pivot floor keeps the value
    finite (and large, via the exploding quadratic form) for near-singular
    R instead of raising.

    Args:
        std_resid: Standardized residuals (T, n)
        Q_bar: Unconditional correlation matrix (n, n)
        a: DCC a parameter
        b: DCC b parameter

    Returns:
        Negative log-likelihood
    """
    T, n = std_resid.shape
    one_minus_ab = 1.0 - a - b

    Q = Q_bar.copy()
    L = np.zeros((n, n))
    y = np.zeros(n)
    d = np.zeros(n)

    ll = 0.0
    for t in range(1, T):
        for i in range(n):
            ei = std_resid[t - 1, i]
            for j in range(n):
                Q[i, j] = one_minus_ab * Q_bar[i, j] + a * ei * std_resid[t - 1, j] + b * Q[i, j]

        for i in range(n):
            qi = Q[i, i]
            if qi < 1e-12:
                qi = 1e-12
            d[i] = 1.0 / np.sqrt(qi)

        # Cholesky of R = D Q D, built element-wise from Q without forming R
        for i in range(n):
            for j in range(i + 1):
                s = Q[i, j] * d[i] * d[j]
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                if i == j:
                    if s < 1e-10:
                        s = 1e-10
                    L[i, i] = np.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]

        logdet = 0.0
        for i in range(n):
            logdet += 2.0 * np.log(L[i, i])

        quad = 0.0
        for i in range(n):
            s = std_resid[t, i]
            for k in range(i):
                s -= L[i, k] * y[k]
            y[i] = s / L[i, i]
            quad += y[i] * y[i]

        ll += logdet + quad

    return 0.5 * ll


@njit(cache=True, nogil=True)
def dcc_covariance_loop(
    std_resid: np.ndarray,
    sigmas: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute DCC correlation and covariance paths.

    Args:
        std_resid: Standardized residuals (T, n)
        sigmas: Conditional standard deviations (T, n)
        Q_bar: Unconditional correlation matrix
        a: DCC a parameter
        b: DCC b parameter

    Returns:
        (R, H, Q_last): R and H of shape (T, n, n), Q_last of shape (n, n)
    """
    T, n = std_resid.shape
    R = np.zeros((T, n, n))
    H = np.zeros((T, n, n))
    d = np.zeros(n)

    Q = Q_bar.copy()
    one_minus_ab = 1.0 - a - b

    for i in range(n):
        for j in range(n):
            R[0, i, j] = Q_bar[i, j]
            H[0, i, j] = sigmas[0, i] * Q_bar[i, j] * sigmas[0, j]

    for t in range(1, T):
        for i in range(n):
            ei = std_resid[t - 1, i]
            for j in range(n):
                Q[i, j] = one_minus_ab * Q_bar[i, j] + a * ei * std_resid[t - 1, j] + b * Q[i, j]

        for i in range(n):
            qi = Q[i, i]
            if qi < 1e-12:
                qi = 1e-12
            d[i] = 1.0 / np.sqrt(qi)

        for i in range(n):
            for j in range(n):
                r = Q[i, j] * d[i] * d[j]
                R[t, i, j] = r
                H[t, i, j] = sigmas[t, i] * r * sigmas[t, j]

    return R, H, Q


@njit(cache=True, nogil=True)
def dcc_final_covariance(
    std_resid: np.ndarray,
    sigmas: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute only the final DCC correlation and covariance matrices.

    Memory-efficient version that doesn't store the full path.

    Args:
        std_resid: Standardized residuals (T, n)
        sigmas: Conditional standard deviations (T, n)
        Q_bar: Unconditional correlation matrix
        a: DCC a parameter
        b: DCC b parameter

    Returns:
        (R_final, H_final, Q_last), each of shape (n, n)
    """
    T, n = std_resid.shape

    Q = Q_bar.copy()
    one_minus_ab = 1.0 - a - b

    for t in range(1, T):
        for i in range(n):
            ei = std_resid[t - 1, i]
            for j in range(n):
                Q[i, j] = one_minus_ab * Q_bar[i, j] + a * ei * std_resid[t - 1, j] + b * Q[i, j]

    R_final = np.zeros((n, n))
    H_final = np.zeros((n, n))
    d = np.zeros(n)
    for i in range(n):
        qi = Q[i, i]
        if qi < 1e-12:
            qi = 1e-12
        d[i] = 1.0 / np.sqrt(qi)
    for i in range(n):
        for j in range(n):
            r = Q[i, j] * d[i] * d[j]
            R_final[i, j] = r
            H_final[i, j] = sigmas[T - 1, i] * r * sigmas[T - 1, j]

    return R_final, H_final, Q
