"""JIT-compiled inner loops for GARCH models."""

import numpy as np
from numba import njit, prange


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
def _garch_nll_grad_into(
    returns: np.ndarray,
    params: np.ndarray,
    p: int,
    q: int,
    var0: float,
    sigma2: np.ndarray,
    ds: np.ndarray,
    grad: np.ndarray,
) -> float:
    """Allocation-free core of ``garch_loglik_grad``.

    Writes the gradient into ``grad`` (length 1 + p + q) using the work
    buffers ``sigma2`` (length T) and ``ds`` (T, 1 + p + q); returns the
    negative log-likelihood.
    """
    T = len(returns)
    k = 1 + p + q
    max_lag = max(p, q)
    omega = params[0]
    if omega < 1e-12:
        omega = 1e-12

    for c in range(k):
        grad[c] = 0.0

    ll = 0.0
    for t in range(min(max_lag, T)):
        sigma2[t] = var0
        for c in range(k):
            ds[t, c] = 0.0
        ll += np.log(var0) + returns[t] ** 2 / var0

    for t in range(max_lag, T):
        s = omega
        for i in range(p):
            s += params[1 + i] * returns[t - 1 - i] ** 2
        for j in range(q):
            s += params[1 + p + j] * sigma2[t - 1 - j]

        r2 = returns[t] ** 2
        if s < 1e-10 or s > 1e12:
            s = 1e-10 if s < 1e-10 else 1e12
            for c in range(k):
                ds[t, c] = 0.0
        else:
            ds[t, 0] = 1.0
            for i in range(p):
                ds[t, 1 + i] = returns[t - 1 - i] ** 2
            for j in range(q):
                ds[t, 1 + p + j] = sigma2[t - 1 - j]
            for j in range(q):
                bj = params[1 + p + j]
                for c in range(k):
                    ds[t, c] += bj * ds[t - 1 - j, c]
            w = 0.5 * (1.0 - r2 / s) / s
            for c in range(k):
                grad[c] += w * ds[t, c]
        sigma2[t] = s
        ll += np.log(s) + r2 / s

    return 0.5 * ll


@njit(cache=True, nogil=True)
def garch_loglik_grad(
    returns: np.ndarray,
    params: np.ndarray,
    p: int,
    q: int,
    var0: float,
) -> tuple[float, np.ndarray]:
    """GARCH(p,q) negative log-likelihood and its analytic gradient.

    Same value as ``garch_loglik``. The gradient propagates
    d sigma2_t / d theta through the recursion
    ``ds_t = (1, r^2_{t-i}, sigma2_{t-j}) + sum_j beta_j ds_{t-j}``;
    steps hitting the variance clamp contribute a zero derivative,
    matching the clamped (locally constant) value.

    Args:
        returns: 1D array of returns
        params: (omega, alpha_1..alpha_p, beta_1..beta_q)
        p: ARCH order
        q: GARCH order
        var0: initial variance estimate

    Returns:
        (nll, grad) with grad of shape (1 + p + q,)
    """
    T = len(returns)
    k = 1 + p + q
    grad = np.empty(k)
    nll = _garch_nll_grad_into(returns, params, p, q, var0, np.empty(T), np.empty((T, k)), grad)
    return nll, grad


# Feasible set for the unit-variance parameters x = (omega, alpha, beta):
# omega >= _OMEGA_MIN, alpha, beta >= 0, sum(alpha) + sum(beta) <= _MAX_PERSISTENCE
_OMEGA_MIN = 1e-12
_MAX_PERSISTENCE = 0.999


@njit(cache=True, nogil=True)
def garch_project(x: np.ndarray, out: np.ndarray) -> None:
    """Euclidean projection of ``x`` onto the GARCH feasible set.

    omega is clipped at its lower bound; the coefficient block is
    clipped at zero and, if its sum still exceeds the persistence cap,
    projected onto the capped simplex ``{y >= 0, sum(y) = cap}``.
    """
    k1 = len(x)
    out[0] = x[0] if x[0] > _OMEGA_MIN else _OMEGA_MIN
    tot = 0.0
    for c in range(1, k1):
        v = x[c] if x[c] > 0.0 else 0.0
        out[c] = v
        tot += v
    if tot <= _MAX_PERSISTENCE:
        return
    # Simplex projection (Held et al.): find tau with sum(max(x - tau, 0)) = cap
    v = np.sort(x[1:])[::-1]
    css = 0.0
    tau = 0.0
    for m in range(k1 - 1):
        css += v[m]
        t = (css - _MAX_PERSISTENCE) / (m + 1)
        if m == k1 - 2 or v[m + 1] <= t:
            tau = t
            break
    for c in range(1, k1):
        d = x[c] - tau
        out[c] = d if d > 0.0 else 0.0


@njit(cache=True, nogil=True)
def _projected_gradient_norm(x: np.ndarray, g: np.ndarray, work: np.ndarray) -> float:
    """max |x - P(x - g)|: zero exactly at a constrained stationary point."""
    k1 = len(x)
    for c in range(k1):
        work[c] = x[c] - g[c]
    garch_project(work.copy(), work)
    m = 0.0
    for c in range(k1):
        d = abs(x[c] - work[c])
        if d > m:
            m = d
    return m


@njit(cache=True, nogil=True)
def garch_fit_projected(
    returns: np.ndarray,
    x0: np.ndarray,
    p: int,
    q: int,
    var0: float,
    maxiter: int,
    gtol: float,
    ftol: float,
) -> tuple[np.ndarray, float, bool]:
    """Minimize the per-observation GARCH NLL over the feasible set.

    Projected BFGS (Kim, Sra & Dhillon 2010): the quasi-Newton direction
    is restricted to free variables — those not held at a bound by the
    gradient — and the step searches along the projection arc
    ``P(x + t d)`` with an Armijo test. Boundary optima (alpha_i = 0,
    persistence at the cap) are reached exactly. The inverse-Hessian
    estimate is reset whenever the free set changes: the free block of
    the full inverse is a poor inverse of the reduced Hessian when
    omega and beta are strongly correlated, and steps zigzag. Converged
    when the projected-gradient norm drops below ``gtol``, or when a
    step improves the objective by less than ``ftol * (1 + |f|)`` (as
    in L-BFGS-B; flat ridges, e.g. beta when alpha = 0).

    Returns:
        (x, nll_per_obs, converged) with x = (omega, alpha, beta)
    """
    T = len(returns)
    k1 = 1 + p + q
    inv_T = 1.0 / T
    sigma2 = np.empty(T)
    ds = np.empty((T, k1))

    x = np.empty(k1)
    garch_project(x0, x)
    g = np.empty(k1)
    x_new = np.empty(k1)
    trial = np.empty(k1)
    g_new = np.empty(k1)
    d = np.empty(k1)
    y = np.empty(k1)
    s = np.empty(k1)
    Hy = np.empty(k1)
    work = np.empty(k1)
    free = np.ones(k1, dtype=np.bool_)
    H = np.eye(k1)
    gamma = 1.0  # last curvature scale sy / yy, for resets
    just_reset = False

    f = _garch_nll_grad_into(returns, x, p, q, var0, sigma2, ds, g) * inv_T
    for c in range(k1):
        g[c] *= inv_T

    converged = False
    scaled = False
    for _ in range(maxiter):
        if _projected_gradient_norm(x, g, work) < gtol:
            converged = True
            break

        # Variables held at a bound by the gradient are fixed this step
        changed = False
        for c in range(k1):
            lower = _OMEGA_MIN if c == 0 else 0.0
            fc = not (x[c] <= lower + 1e-14 and g[c] > 0.0)
            if fc != free[c]:
                changed = True
            free[c] = fc
        if changed:
            for c in range(k1):
                for e in range(k1):
                    H[c, e] = gamma if c == e else 0.0

        for c in range(k1):
            acc = 0.0
            if free[c]:
                for e in range(k1):
                    if free[e]:
                        acc -= H[c, e] * g[e]
            d[c] = acc

        # Persistence at the cap: keep the step on the face sum(d_coef) = 0
        # (null-space projection in the H metric) when it would push out
        coef_sum = 0.0
        ad = 0.0
        for c in range(1, k1):
            coef_sum += x[c]
            if free[c]:
                ad += d[c]
        if coef_sum >= _MAX_PERSISTENCE - 1e-12 and ad > 0.0:
            aHa = 0.0
            for c in range(1, k1):
                acc = 0.0
                if free[c]:
                    for e in range(1, k1):
                        if free[e]:
                            acc += H[c, e]
                work[c] = acc  # (H a)_c on free coefficients
                aHa += acc if free[c] else 0.0
            if aHa > 0.0:
                for c in range(1, k1):
                    if free[c]:
                        d[c] -= work[c] * ad / aHa
            # omega's coupling through H: include it in the same projection
            acc = 0.0
            if free[0]:
                for e in range(1, k1):
                    if free[e]:
                        acc += H[0, e]
                if aHa > 0.0:
                    d[0] -= acc * ad / aHa

        # Search along the projection arc; fall back to steepest descent
        ok = False
        for attempt in range(2):
            step = 1.0
            for _ls in range(40):
                for c in range(k1):
                    trial[c] = x[c] + step * d[c]
                garch_project(trial, x_new)
                decrease = 0.0
                for c in range(k1):
                    decrease += g[c] * (x_new[c] - x[c])
                if decrease < 0.0:
                    f_new = (
                        _garch_nll_grad_into(returns, x_new, p, q, var0, sigma2, ds, g_new) * inv_T
                    )
                    if f_new <= f + 1e-4 * decrease:
                        ok = True
                        break
                step *= 0.5
            if ok or attempt == 1:
                break
            for c in range(k1):
                d[c] = -g[c]
                for e in range(k1):
                    H[c, e] = 1.0 if c == e else 0.0
            scaled = False
        if not ok:
            converged = _projected_gradient_norm(x, g, work) < 10.0 * gtol
            break

        # Curvature pairs on the free coordinates only
        sy = 0.0
        yy = 0.0
        for c in range(k1):
            g_new[c] *= inv_T
            s[c] = x_new[c] - x[c]
            y[c] = g_new[c] - g[c] if free[c] else 0.0
            sy += s[c] * y[c]
            yy += y[c] * y[c]
        df = f - f_new
        for c in range(k1):
            x[c] = x_new[c]
            g[c] = g_new[c]
        f = f_new
        if df <= ftol * (1.0 + abs(f)):
            pg = _projected_gradient_norm(x, g, work)
            if pg < 1e3 * gtol:
                converged = True
                break
            if just_reset:  # stalled even from a fresh H: give up
                break
            for c in range(k1):
                for e in range(k1):
                    H[c, e] = gamma if c == e else 0.0
            just_reset = True
            continue
        just_reset = False

        if sy > 1e-12 * yy and sy > 0.0:
            gamma = sy / yy
            if not scaled:  # Nocedal-Wright initial scaling of H
                for c in range(k1):
                    for e in range(k1):
                        H[c, e] = gamma if c == e else 0.0
                scaled = True
            yHy = 0.0
            for c in range(k1):
                acc = 0.0
                for e in range(k1):
                    acc += H[c, e] * y[e]
                Hy[c] = acc
                yHy += y[c] * acc
            rho = 1.0 / sy
            for c in range(k1):
                for e in range(k1):
                    H[c, e] += (1.0 + yHy * rho) * rho * s[c] * s[e] - rho * (
                        Hy[c] * s[e] + s[c] * Hy[e]
                    )

    return x, f, converged


@njit(cache=True, nogil=True)
def _on_boundary(x: np.ndarray) -> bool:
    """True if a coefficient is at (or within 1e-3 of) 0, or persistence is at the cap."""
    tot = 0.0
    for c in range(1, len(x)):
        if x[c] < 1e-3:
            return True
        tot += x[c]
    return tot >= _MAX_PERSISTENCE - 1e-12


@njit(cache=True, nogil=True, parallel=True)
def garch_fit_batch(
    Z: np.ndarray,
    X0: np.ndarray,
    p: int,
    q: int,
    var0: np.ndarray,
    maxiter: int,
    gtol: float,
    ftol: float,
    n_always: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit GARCH(p,q) to each row of ``Z`` (n, T) in parallel.

    Runs ``garch_fit_projected`` from the first ``n_always`` starts in
    ``X0`` (n_starts, 1+p+q), and from the rest only while the best fit
    so far sits on or near the boundary (a coefficient below 1e-3 or
    persistence at the cap) — the signature of a flat ridge the start led into. Keeps
    the lowest objective. Rows are independent, so results do not
    depend on the thread count.

    Returns:
        (params (n, 1+p+q), nll_per_obs (n,), converged (n,) as uint8)
    """
    n = Z.shape[0]
    k1 = 1 + p + q
    out = np.empty((n, k1))
    nll = np.empty(n)
    conv = np.zeros(n, dtype=np.uint8)
    for i in prange(n):
        best = np.inf
        for st in range(X0.shape[0]):
            if st >= n_always and not _on_boundary(out[i]):
                break
            x, f, ok = garch_fit_projected(Z[i], X0[st], p, q, var0[i], maxiter, gtol, ftol)
            if f < best:
                best = f
                for c in range(k1):
                    out[i, c] = x[c]
                conv[i] = 1 if ok else 0
        nll[i] = best
    return out, nll, conv


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


@njit(cache=True, nogil=True, parallel=True)
def dcc_cl_loglik(
    std_resid: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> float:
    """Pairwise composite negative log-likelihood over contiguous pairs.

    Sums closed-form bivariate Gaussian NLLs (0.5*(log(1-rho^2)+quad), t>=1)
    over pairs (i, i+1). O(T*n) per evaluation, no linear algebra. The
    correlation is clamped to +/-(1 - 1e-8) so the value is finite
    everywhere.

    Args:
        std_resid: Standardized residuals (T, n)
        Q_bar: Unconditional correlation matrix (n, n)
        a: DCC a parameter
        b: DCC b parameter

    Returns:
        Composite negative log-likelihood
    """
    T, n = std_resid.shape
    one_minus_ab = 1.0 - a - b
    rho_max = 1.0 - 1e-8

    total = 0.0
    for k in prange(n - 1):
        i = k
        j = k + 1
        qbar_ij = Q_bar[i, j]
        # Q_bar has a unit diagonal, so the diagonal recursions use
        # one_minus_ab directly
        qii = 1.0
        qjj = 1.0
        qij = qbar_ij
        ll = 0.0
        for t in range(1, T):
            ei = std_resid[t - 1, i]
            ej = std_resid[t - 1, j]
            qii = one_minus_ab + a * ei * ei + b * qii
            qjj = one_minus_ab + a * ej * ej + b * qjj
            qij = one_minus_ab * qbar_ij + a * ei * ej + b * qij

            denom = qii * qjj
            if denom < 1e-24:
                denom = 1e-24
            rho = qij / np.sqrt(denom)
            if rho > rho_max:
                rho = rho_max
            elif rho < -rho_max:
                rho = -rho_max

            one_m_r2 = 1.0 - rho * rho
            xi = std_resid[t, i]
            xj = std_resid[t, j]
            ll += np.log(one_m_r2) + (xi * xi + xj * xj - 2.0 * rho * xi * xj) / one_m_r2
        total += ll

    return 0.5 * total


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


@njit(cache=True, nogil=True, parallel=True)
def dcc_final_covariance(
    std_resid: np.ndarray,
    sigmas: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute only the final DCC correlation and covariance matrices.

    Memory-efficient version that doesn't store the full path. Each
    Q[i, j] follows its own scalar recursion, so rows of the lower
    triangle (Q is symmetric) run in parallel, paired (k, n-1-k) to
    balance work; within a row the update is vectorized across j.

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
    one_minus_ab = 1.0 - a - b

    Q = np.empty((n, n))
    for k in prange((n + 1) // 2):
        c = np.empty(n)
        qrow = np.empty(n)
        for r in range(2):
            i = k + r * (n - 1 - 2 * k)
            if r == 1 and i == k:  # middle row of an odd n
                continue
            m = i + 1
            for j in range(m):
                c[j] = one_minus_ab * Q_bar[i, j]
                qrow[j] = Q_bar[i, j]
            for t in range(1, T):
                ai = a * std_resid[t - 1, i]
                for j in range(m):
                    qrow[j] = c[j] + ai * std_resid[t - 1, j] + b * qrow[j]
            for j in range(m):
                Q[i, j] = qrow[j]
                Q[j, i] = qrow[j]

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
