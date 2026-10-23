"""JIT-compiled inner loops for GARCH models."""

import math

import numpy as np
from numba import njit, prange

# Parameter vectors for the univariate kernels are laid out as
#   x = (omega, alpha_1..alpha_p, g_1..g_o, beta_1..beta_q[, eta])
# in the unit-variance parametrization. g = gamma / 2 is the GJR
# asymmetry term, so stationarity (alpha + beta + gamma/2 under symmetric
# innovations) is a plain sum over the coefficient block x[1 : 1+p+o+q].
# eta = 1 / nu is present only for Student-t innovations (dist == 1).
DIST_NORMAL = 0
DIST_T = 1

# Feasible set: omega >= _OMEGA_MIN, coefficients >= 0, their sum
# <= _MAX_PERSISTENCE, eta in [_ETA_MIN, _ETA_MAX] (nu in [2.1, 200])
_OMEGA_MIN = 1e-12
_MAX_PERSISTENCE = 0.999
_ETA_MIN = 1.0 / 200.0
_ETA_MAX = 1.0 / 2.1


@njit(cache=True, nogil=True)
def digamma(x: float) -> float:
    """Digamma function for x > 0 (recurrence to x >= 10, then asymptotic series)."""
    r = 0.0
    while x < 10.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    series = f * (
        1.0 / 12.0 - f * (1.0 / 120.0 - f * (1.0 / 252.0 - f * (1.0 / 240.0 - f / 132.0)))
    )
    return r + np.log(x) - 0.5 / x - series


@njit(cache=True, nogil=True)
def garch_variance_loop(
    returns: np.ndarray,
    params: np.ndarray,
    p: int,
    o: int,
    q: int,
    var0: float,
) -> np.ndarray:
    """Compute the GJR-GARCH(p,o,q) conditional variance series.

    ``sigma2_t = omega + sum_i alpha_i r_{t-i}^2
    + sum_k 2 g_k r_{t-k}^2 1[r_{t-k} < 0] + sum_j beta_j sigma2_{t-j}``.
    Each step is clamped to [1e-10, 1e12] so the recursion (and any
    likelihood built on it) stays finite even at infeasible parameters
    probed by the optimizer.

    Args:
        returns: 1D array of returns
        params: (omega, alpha, g, beta[, eta]); eta is ignored
        p: ARCH order
        o: asymmetric (GJR) order
        q: GARCH order
        var0: initial variance estimate

    Returns:
        Conditional variance series
    """
    T = len(returns)
    max_lag = max(p, o, q)
    omega = params[0]
    if omega < _OMEGA_MIN:
        omega = _OMEGA_MIN
    sigma2 = np.full(T, var0)
    for t in range(max_lag, T):
        s = omega
        for i in range(p):
            s += params[1 + i] * returns[t - 1 - i] ** 2
        for k in range(o):
            r = returns[t - 1 - k]
            if r < 0.0:
                s += 2.0 * params[1 + p + k] * r * r
        for j in range(q):
            s += params[1 + p + o + j] * sigma2[t - 1 - j]
        if s < 1e-10:
            s = 1e-10
        elif s > 1e12:
            s = 1e12
        sigma2[t] = s
    return sigma2


@njit(cache=True, nogil=True)
def _garch_nll_grad_into(
    returns: np.ndarray,
    params: np.ndarray,
    p: int,
    o: int,
    q: int,
    dist: int,
    var0: float,
    sigma2: np.ndarray,
    ds: np.ndarray,
    grad: np.ndarray,
) -> float:
    """Negative log-likelihood and analytic gradient (allocation-free).

    Normal: ``0.5 * sum(log s + r^2 / s)`` (the 2*pi constant is left to
    the caller). Student-t, standardized to unit variance, with
    ``nu = 1 / eta``: ``sum(-c(nu) + 0.5 log s + (nu+1)/2 log(1 + r^2 /
    ((nu-2) s)))``, ``c(nu) = lgamma((nu+1)/2) - lgamma(nu/2) -
    0.5 log(pi (nu-2))`` — the full density, constants included.

    The gradient propagates d sigma2_t / d theta through the recursion;
    steps hitting the variance clamp contribute a zero variance
    derivative. Writes the gradient into ``grad`` (len(params)) using
    the work buffers ``sigma2`` (T,) and ``ds`` (T, 1+p+o+q).
    """
    T = len(returns)
    kv = 1 + p + o + q  # parameters entering the variance recursion
    k1 = kv + (1 if dist == DIST_T else 0)
    max_lag = max(p, o, q)
    omega = params[0]
    if omega < _OMEGA_MIN:
        omega = _OMEGA_MIN

    for c in range(k1):
        grad[c] = 0.0

    nu = 0.0
    nu_m2 = 0.0
    half_nu1 = 0.0
    if dist == DIST_T:
        nu = 1.0 / params[kv]
        nu_m2 = nu - 2.0
        half_nu1 = 0.5 * (nu + 1.0)
        const = math.lgamma(half_nu1) - math.lgamma(0.5 * nu) - 0.5 * np.log(np.pi * nu_m2)
        dconst = 0.5 * digamma(half_nu1) - 0.5 * digamma(0.5 * nu) - 0.5 / nu_m2
    else:
        const = 0.0
        dconst = 0.0
    dnu = 0.0  # accumulated d nll / d nu

    ll = 0.0
    for t in range(T):
        r2 = returns[t] ** 2
        if t < max_lag:
            s = var0
            for c in range(kv):
                ds[t, c] = 0.0
            clamped = True
        else:
            s = omega
            for i in range(p):
                s += params[1 + i] * returns[t - 1 - i] ** 2
            for k in range(o):
                r = returns[t - 1 - k]
                if r < 0.0:
                    s += 2.0 * params[1 + p + k] * r * r
            for j in range(q):
                s += params[1 + p + o + j] * sigma2[t - 1 - j]
            clamped = s < 1e-10 or s > 1e12
            if clamped:
                s = 1e-10 if s < 1e-10 else 1e12
                for c in range(kv):
                    ds[t, c] = 0.0
            else:
                ds[t, 0] = 1.0
                for i in range(p):
                    ds[t, 1 + i] = returns[t - 1 - i] ** 2
                for k in range(o):
                    r = returns[t - 1 - k]
                    ds[t, 1 + p + k] = 2.0 * r * r if r < 0.0 else 0.0
                for j in range(q):
                    ds[t, 1 + p + o + j] = sigma2[t - 1 - j]
                for j in range(q):
                    bj = params[1 + p + o + j]
                    for c in range(kv):
                        ds[t, c] += bj * ds[t - 1 - j, c]
        sigma2[t] = s

        if dist == DIST_T:
            u = r2 / (nu_m2 * s)
            l1u = np.log1p(u)
            ll += -const + 0.5 * np.log(s) + half_nu1 * l1u
            dnu += -dconst + 0.5 * l1u - half_nu1 * u / ((1.0 + u) * nu_m2)
            w = (0.5 - half_nu1 * u / (1.0 + u)) / s
        else:
            ll += 0.5 * (np.log(s) + r2 / s)
            w = 0.5 * (1.0 - r2 / s) / s
        if not clamped:
            for c in range(kv):
                grad[c] += w * ds[t, c]

    if dist == DIST_T:
        grad[kv] = -dnu * nu * nu  # d/d eta = d/d nu * (-nu^2)
    return ll


@njit(cache=True, nogil=True)
def garch_loglik_grad(
    returns: np.ndarray,
    params: np.ndarray,
    p: int,
    o: int,
    q: int,
    dist: int,
    var0: float,
) -> tuple[float, np.ndarray]:
    """GJR-GARCH negative log-likelihood and its analytic gradient.

    Args:
        returns: 1D array of returns
        params: (omega, alpha, g, beta[, eta]), see the module notes
        p, o, q: ARCH, asymmetric and GARCH orders
        dist: DIST_NORMAL or DIST_T
        var0: initial variance estimate

    Returns:
        (nll, grad) with grad of shape params.shape
    """
    T = len(returns)
    grad = np.empty(len(params))
    nll = _garch_nll_grad_into(
        returns, params, p, o, q, dist, var0, np.empty(T), np.empty((T, 1 + p + o + q)), grad
    )
    return nll, grad


@njit(cache=True, nogil=True)
def garch_loglik(
    returns: np.ndarray,
    params: np.ndarray,
    p: int,
    o: int,
    q: int,
    dist: int,
    var0: float,
) -> float:
    """GJR-GARCH negative log-likelihood (value of ``garch_loglik_grad``)."""
    return garch_loglik_grad(returns, params, p, o, q, dist, var0)[0]


@njit(cache=True, nogil=True)
def garch_project(x: np.ndarray, out: np.ndarray, n_coef: int) -> None:
    """Euclidean projection of ``x`` onto the GARCH feasible set.

    omega is clipped at its lower bound; the coefficient block
    ``x[1 : 1+n_coef]`` is clipped at zero and, if its sum still exceeds
    the persistence cap, projected onto the capped simplex
    ``{y >= 0, sum(y) = cap}``; a trailing eta is clipped to its box.
    """
    k1 = len(x)
    out[0] = x[0] if x[0] > _OMEGA_MIN else _OMEGA_MIN
    tot = 0.0
    for c in range(1, 1 + n_coef):
        v = x[c] if x[c] > 0.0 else 0.0
        out[c] = v
        tot += v
    for c in range(1 + n_coef, k1):
        out[c] = min(max(x[c], _ETA_MIN), _ETA_MAX)
    if tot <= _MAX_PERSISTENCE:
        return
    # Simplex projection (Held et al.): find tau with sum(max(x - tau, 0)) = cap
    v = np.sort(x[1 : 1 + n_coef])[::-1]
    css = 0.0
    tau = 0.0
    for m in range(n_coef):
        css += v[m]
        t = (css - _MAX_PERSISTENCE) / (m + 1)
        if m == n_coef - 1 or v[m + 1] <= t:
            tau = t
            break
    for c in range(1, 1 + n_coef):
        d = x[c] - tau
        out[c] = d if d > 0.0 else 0.0


@njit(cache=True, nogil=True)
def _projected_gradient_norm(x: np.ndarray, g: np.ndarray, work: np.ndarray, n_coef: int) -> float:
    """max |x - P(x - g)|: zero exactly at a constrained stationary point."""
    k1 = len(x)
    for c in range(k1):
        work[c] = x[c] - g[c]
    garch_project(work.copy(), work, n_coef)
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
    o: int,
    q: int,
    dist: int,
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
    gamma_k = 0, persistence at the cap) are reached exactly. The
    inverse-Hessian estimate is reset whenever the free set changes: the
    free block of the full inverse is a poor inverse of the reduced
    Hessian when omega and beta are strongly correlated, and steps
    zigzag. Converged when the projected-gradient norm drops below
    ``gtol``, or when a step improves the objective by less than
    ``ftol * (1 + |f|)`` (as in L-BFGS-B; flat ridges, e.g. beta when
    alpha = 0).

    Returns:
        (x, nll_per_obs, converged) with x laid out as in the module notes
    """
    T = len(returns)
    n_coef = p + o + q
    kv = 1 + n_coef
    k1 = kv + (1 if dist == DIST_T else 0)
    inv_T = 1.0 / T
    sigma2 = np.empty(T)
    ds = np.empty((T, kv))

    x = np.empty(k1)
    garch_project(x0, x, n_coef)
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

    f = _garch_nll_grad_into(returns, x, p, o, q, dist, var0, sigma2, ds, g) * inv_T
    for c in range(k1):
        g[c] *= inv_T

    converged = False
    scaled = False
    for _ in range(maxiter):
        if _projected_gradient_norm(x, g, work, n_coef) < gtol:
            converged = True
            break

        # Variables held at a bound by the gradient are fixed this step
        changed = False
        for c in range(k1):
            if c == 0:
                held = x[c] <= _OMEGA_MIN + 1e-14 and g[c] > 0.0
            elif c < kv:
                held = x[c] <= 1e-14 and g[c] > 0.0
            else:
                held = (x[c] <= _ETA_MIN + 1e-14 and g[c] > 0.0) or (
                    x[c] >= _ETA_MAX - 1e-14 and g[c] < 0.0
                )
            fc = not held
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
        for c in range(1, kv):
            coef_sum += x[c]
            if free[c]:
                ad += d[c]
        if coef_sum >= _MAX_PERSISTENCE - 1e-12 and ad > 0.0:
            aHa = 0.0
            for c in range(k1):
                acc = 0.0
                if free[c]:
                    for e in range(1, kv):
                        if free[e]:
                            acc += H[c, e]
                work[c] = acc  # (H a)_c, a = indicator of free coefficients
                if 1 <= c < kv and free[c]:
                    aHa += acc
            if aHa > 0.0:
                for c in range(k1):
                    if free[c]:
                        d[c] -= work[c] * ad / aHa

        # Search along the projection arc; fall back to steepest descent
        ok = False
        for attempt in range(2):
            step = 1.0
            for _ls in range(40):
                for c in range(k1):
                    trial[c] = x[c] + step * d[c]
                garch_project(trial, x_new, n_coef)
                decrease = 0.0
                for c in range(k1):
                    decrease += g[c] * (x_new[c] - x[c])
                if decrease < 0.0:
                    f_new = (
                        _garch_nll_grad_into(returns, x_new, p, o, q, dist, var0, sigma2, ds, g_new)
                        * inv_T
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
            converged = _projected_gradient_norm(x, g, work, n_coef) < 10.0 * gtol
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
            pg = _projected_gradient_norm(x, g, work, n_coef)
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
def _on_boundary(x: np.ndarray, n_coef: int) -> bool:
    """True if a coefficient is below 1e-3 or persistence is at the cap."""
    tot = 0.0
    for c in range(1, 1 + n_coef):
        if x[c] < 1e-3:
            return True
        tot += x[c]
    return tot >= _MAX_PERSISTENCE - 1e-12


@njit(cache=True, nogil=True, parallel=True)
def garch_fit_batch(
    Z: np.ndarray,
    X0: np.ndarray,
    p: int,
    o: int,
    q: int,
    dist: int,
    var0: np.ndarray,
    maxiter: int,
    gtol: float,
    ftol: float,
    n_always: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit GJR-GARCH(p,o,q) to each row of ``Z`` (n, T) in parallel.

    Runs ``garch_fit_projected`` from the first ``n_always`` starts in
    ``X0`` (n_starts, n_params), and from the rest only while the best
    fit so far sits on or near the boundary (a coefficient below 1e-3 or
    persistence at the cap) — the signature of a flat ridge the start
    led into. Keeps the lowest objective. Rows are independent, so
    results do not depend on the thread count.

    Returns:
        (params (n, n_params), nll_per_obs (n,), converged (n,) as uint8)
    """
    n = Z.shape[0]
    k1 = X0.shape[1]
    n_coef = p + o + q
    out = np.empty((n, k1))
    nll = np.empty(n)
    conv = np.zeros(n, dtype=np.uint8)
    for i in prange(n):
        best = np.inf
        for st in range(X0.shape[0]):
            if st >= n_always and not _on_boundary(out[i], n_coef):
                break
            x, f, ok = garch_fit_projected(
                Z[i], X0[st], p, o, q, dist, var0[i], maxiter, gtol, ftol
            )
            if f < best:
                best = f
                for c in range(k1):
                    out[i, c] = x[c]
                conv[i] = 1 if ok else 0
        nll[i] = best
    return out, nll, conv


# dcc_loglik_loop splits time into this many parallel blocks (fixed, so
# results are independent of the thread count), keeping a Q snapshot per
# block start; snapshots are capped in total, which lowers the count at
# large n
_DCC_TIME_BLOCKS = 64
_DCC_SNAPSHOT_BYTES = 64 * 1024 * 1024


@njit(cache=True, nogil=True)
def _dcc_nll_block(
    std_resid: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
    Q: np.ndarray,
    t0: int,
    t1: int,
) -> float:
    """Sum of log|R_t| + e_t' R_t^-1 e_t over t in [t0, t1).

    ``Q`` holds Q_{t0-1} on entry and is advanced in place. The Cholesky
    pivot floor keeps the value finite (and large, via the exploding
    quadratic form) for near-singular R instead of raising.
    """
    n = std_resid.shape[1]
    one_minus_ab = 1.0 - a - b
    L = np.zeros((n, n))
    y = np.zeros(n)
    d = np.zeros(n)

    ll = 0.0
    for t in range(t0, t1):
        for i in range(n):
            ei = std_resid[t - 1, i]
            for j in range(n):
                Q[i, j] = one_minus_ab * Q_bar[i, j] + a * ei * std_resid[t - 1, j] + b * Q[i, j]

        for i in range(n):
            qi = Q[i, i]
            if qi < 1e-12:
                qi = 1e-12
            d[i] = 1.0 / np.sqrt(qi)

        # Cholesky of R = D Q D, built element-wise from Q without forming
        # R. Four accumulators give the row dot products instruction-level
        # parallelism (a single running sum cannot vectorize without
        # fastmath).
        for i in range(n):
            for j in range(i + 1):
                acc0 = 0.0
                acc1 = 0.0
                acc2 = 0.0
                acc3 = 0.0
                k = 0
                while k + 4 <= j:
                    acc0 += L[i, k] * L[j, k]
                    acc1 += L[i, k + 1] * L[j, k + 1]
                    acc2 += L[i, k + 2] * L[j, k + 2]
                    acc3 += L[i, k + 3] * L[j, k + 3]
                    k += 4
                while k < j:
                    acc0 += L[i, k] * L[j, k]
                    k += 1
                s = Q[i, j] * d[i] * d[j] - ((acc0 + acc1) + (acc2 + acc3))
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

    return ll


@njit(cache=True, nogil=True, parallel=True)
def dcc_loglik_loop(
    std_resid: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> float:
    """DCC negative log-likelihood: 0.5 * sum_{t>=1}(log|R_t| + e' R_t^-1 e).

    The per-step O(n^3) Cholesky work is independent across t once Q_t
    is known, so time is split into ``_DCC_TIME_BLOCKS`` blocks run in
    parallel — a fixed count, so the result does not depend on the
    thread count. A first pass
    runs the cheap O(n^2) Q recursion (parallel over rows — each element
    has its own scalar recursion) and snapshots Q at the block starts;
    snapshots are capped at ``_DCC_SNAPSHOT_BYTES``, which bounds the
    block count at large n. Q values match the sequential recursion
    exactly; summation order differs from a sequential pass, which only
    matters (beyond ~1e-10 relative) for singular Q_bar, where the pivot
    floor makes the value huge and rounding-sensitive anyway.

    Args:
        std_resid: Standardized residuals (T, n)
        Q_bar: Unconditional correlation matrix (n, n)
        a: DCC a parameter
        b: DCC b parameter

    Returns:
        Negative log-likelihood
    """
    T, n = std_resid.shape
    if T < 2:
        return 0.0
    one_minus_ab = 1.0 - a - b

    nb = min(_DCC_TIME_BLOCKS, T - 1, max(1, _DCC_SNAPSHOT_BYTES // (8 * n * n)))
    starts = np.empty(nb + 1, dtype=np.int64)
    for k in range(nb + 1):
        starts[k] = 1 + ((T - 1) * k) // nb

    # Pass 1: snaps[k] = Q_{starts[k] - 1}
    snaps = np.empty((nb, n, n))
    for i in prange(n):
        c = np.empty(n)
        q = np.empty(n)
        for j in range(n):
            c[j] = one_minus_ab * Q_bar[i, j]
            q[j] = Q_bar[i, j]
        k = 0
        for t in range(0, starts[nb - 1]):
            if t > 0:
                ai = a * std_resid[t - 1, i]
                for j in range(n):
                    q[j] = c[j] + ai * std_resid[t - 1, j] + b * q[j]
            while k < nb and starts[k] - 1 == t:
                for j in range(n):
                    snaps[k, i, j] = q[j]
                k += 1

    # Pass 2: Cholesky work per time block
    parts = np.empty(nb)
    for k in prange(nb):
        parts[k] = _dcc_nll_block(std_resid, Q_bar, a, b, snaps[k], starts[k], starts[k + 1])

    ll = 0.0
    for k in range(nb):
        ll += parts[k]
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
