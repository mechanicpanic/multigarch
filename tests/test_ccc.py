import numpy as np
import pytest

from multigarch import CCC


def test_shrinkage_one_gives_identity():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((300, 4)) * 0.01
    m = CCC(n_jobs=1, shrinkage=1.0).fit(returns)
    np.testing.assert_allclose(m.R, np.eye(4), atol=1e-12)


def test_shrinkage_interpolates():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((300, 4)) * 0.01
    m0 = CCC(n_jobs=1, shrinkage=0.0).fit(returns)
    m5 = CCC(n_jobs=1, shrinkage=0.5).fit(returns)
    np.testing.assert_allclose(m5.R, 0.5 * m0.R + 0.5 * np.eye(4), atol=1e-10)


def test_invalid_shrinkage_raises():
    with pytest.raises(ValueError, match="shrinkage"):
        CCC(shrinkage=1.5)


def _fit_ccc(seed=0, T=300, n=4):
    rng = np.random.default_rng(seed)
    return CCC(n_jobs=1).fit(rng.standard_normal((T, n)) * 0.01)


def test_h_path_matches_reference():
    m = _fit_ccc()
    sigmas = np.sqrt(np.column_stack([g.sigma2 for g in m.garch_models]))
    for t in [0, 150, 299]:
        D = np.diag(sigmas[t])
        np.testing.assert_allclose(m.H[t], D @ m.R @ D, rtol=1e-10)


def test_r_properties():
    m = _fit_ccc()
    np.testing.assert_allclose(np.diag(m.R), 1.0, atol=1e-10)
    np.testing.assert_allclose(m.R, m.R.T, atol=1e-12)
    assert np.linalg.eigvalsh(m.R).min() > -1e-10


def test_low_memory_matches_final_step():
    rng = np.random.default_rng(1)
    returns = rng.standard_normal((300, 4)) * 0.01
    full = CCC(n_jobs=1).fit(returns)
    lm = CCC(n_jobs=1, low_memory=True).fit(returns)
    np.testing.assert_allclose(lm.H, full.H[-1], rtol=1e-10)


def test_forecast_shapes_and_symmetry():
    m = _fit_ccc()
    f = m.forecast(5)
    assert f.shape == (5, 4, 4)
    for h in range(5):
        np.testing.assert_allclose(f[h], f[h].T, atol=1e-12)
        assert np.linalg.eigvalsh(f[h]).min() > 0


def test_single_asset():
    rng = np.random.default_rng(2)
    m = CCC(n_jobs=1).fit(rng.standard_normal(200) * 0.01)
    assert m.H.shape == (200, 1, 1)
    assert np.all(np.isfinite(m.H))
