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
