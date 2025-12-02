"""JIT-compiled inner loops for GARCH models."""

import numpy as np
from numba import njit


@njit(cache=True)
def garch_variance_loop(
    returns: np.ndarray,
    omega: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    var0: float,
) -> np.ndarray:
    """Compute GARCH(p,q) conditional variance series.

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
        arch_term = 0.0
        for i in range(p):
            arch_term += alpha[i] * returns[t - 1 - i] ** 2

        garch_term = 0.0
        for j in range(q):
            garch_term += beta[j] * sigma2[t - 1 - j]

        sigma2[t] = omega + arch_term + garch_term

    return np.maximum(sigma2, 1e-10)


@njit(cache=True)
def garch_loglik(
    returns: np.ndarray,
    omega: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    var0: float,
) -> float:
    """Compute GARCH(p,q) log-likelihood.

    Args:
        returns: 1D array of returns
        omega: intercept
        alpha: ARCH coefficients
        beta: GARCH coefficients
        var0: initial variance estimate

    Returns:
        Negative log-likelihood (for minimization)
    """
    if omega <= 0:
        return 1e10

    for i in range(len(alpha)):
        if alpha[i] < 0:
            return 1e10

    for j in range(len(beta)):
        if beta[j] < 0:
            return 1e10

    persistence = 0.0
    for i in range(len(alpha)):
        persistence += alpha[i]
    for j in range(len(beta)):
        persistence += beta[j]

    if persistence >= 1.0:
        return 1e10

    sigma2 = garch_variance_loop(returns, omega, alpha, beta, var0)

    ll = 0.0
    for t in range(len(returns)):
        ll += np.log(sigma2[t]) + returns[t] ** 2 / sigma2[t]

    return 0.5 * ll


@njit(cache=True)
def dcc_loglik_loop(
    std_resid: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> float:
    """Compute DCC log-likelihood.

    Args:
        std_resid: Standardized residuals (T, n)
        Q_bar: Unconditional correlation matrix (n, n)
        a: DCC a parameter
        b: DCC b parameter

    Returns:
        Negative log-likelihood
    """
    if a < 0 or b < 0 or a + b >= 1:
        return 1e10

    T, n = std_resid.shape
    Q = Q_bar.copy()
    ll = 0.0
    one_minus_ab = 1.0 - a - b

    for t in range(1, T):
        # Update Q
        eps = std_resid[t - 1, :].reshape(-1, 1)
        Q = one_minus_ab * Q_bar + a * (eps @ eps.T) + b * Q

        # Normalize to correlation matrix R
        Q_diag = np.diag(Q).copy()
        for i in range(n):
            if Q_diag[i] <= 0:
                return 1e10
            Q_diag[i] = 1.0 / np.sqrt(Q_diag[i])

        R = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                R[i, j] = Q[i, j] * Q_diag[i] * Q_diag[j]

        # Log-likelihood: -0.5 * (log|R| + eps' R^{-1} eps)
        # Use Cholesky for numerical stability
        try:
            L = np.linalg.cholesky(R)
        except:
            return 1e10

        logdet = 0.0
        for i in range(n):
            logdet += 2.0 * np.log(L[i, i])

        eps_t = std_resid[t, :].copy()

        # Solve L @ y = eps_t
        y = np.zeros(n)
        for i in range(n):
            s = eps_t[i]
            for j in range(i):
                s -= L[i, j] * y[j]
            y[i] = s / L[i, i]

        quad_form = 0.0
        for i in range(n):
            quad_form += y[i] ** 2

        ll += logdet + quad_form

    return 0.5 * ll


@njit(cache=True)
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
        Tuple of (R, H) arrays of shape (T, n, n)
    """
    T, n = std_resid.shape
    R = np.zeros((T, n, n))
    H = np.zeros((T, n, n))

    Q = Q_bar.copy()
    R[0] = Q_bar.copy()

    # First covariance matrix
    for i in range(n):
        for j in range(n):
            H[0, i, j] = sigmas[0, i] * R[0, i, j] * sigmas[0, j]

    one_minus_ab = 1.0 - a - b

    for t in range(1, T):
        eps = std_resid[t - 1, :].reshape(-1, 1)
        Q = one_minus_ab * Q_bar + a * (eps @ eps.T) + b * Q

        # Normalize to correlation
        Q_diag_inv_sqrt = np.zeros(n)
        for i in range(n):
            Q_diag_inv_sqrt[i] = 1.0 / np.sqrt(Q[i, i])

        for i in range(n):
            for j in range(n):
                R[t, i, j] = Q[i, j] * Q_diag_inv_sqrt[i] * Q_diag_inv_sqrt[j]

        # Covariance
        for i in range(n):
            for j in range(n):
                H[t, i, j] = sigmas[t, i] * R[t, i, j] * sigmas[t, j]

    return R, H, Q


@njit(cache=True)
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
        Tuple of (R_final, H_final) arrays of shape (n, n)
    """
    T, n = std_resid.shape

    Q = Q_bar.copy()
    one_minus_ab = 1.0 - a - b

    for t in range(1, T):
        eps = std_resid[t - 1, :].reshape(-1, 1)
        Q = one_minus_ab * Q_bar + a * (eps @ eps.T) + b * Q

    # Final correlation matrix
    R_final = np.zeros((n, n))
    Q_diag_inv_sqrt = np.zeros(n)
    for i in range(n):
        Q_diag_inv_sqrt[i] = 1.0 / np.sqrt(Q[i, i])

    for i in range(n):
        for j in range(n):
            R_final[i, j] = Q[i, j] * Q_diag_inv_sqrt[i] * Q_diag_inv_sqrt[j]

    # Final covariance matrix
    H_final = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H_final[i, j] = sigmas[-1, i] * R_final[i, j] * sigmas[-1, j]

    return R_final, H_final, Q
