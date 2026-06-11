"""Simple multivariate GARCH models for Python.

Univariate `GARCH(p,q)` plus two multivariate correlation models —
`CCC` (constant conditional correlation) and `DCC` (dynamic conditional
correlation) — with numba-JIT inner loops. DCC scales to hundreds of
assets via a pairwise composite-likelihood estimator (`method="cl"`,
chosen automatically for n > 25).

Quickstart:

```python
import numpy as np
from multigarch import GARCH, CCC, DCC

returns = np.random.randn(2000, 100) * 0.02   # (T, n_assets)

dcc = DCC(low_memory=True).fit(returns)
dcc.a, dcc.b          # correlation dynamics parameters
dcc.H                 # conditional covariance (final, or (T,n,n) path)
dcc.forecast(5)       # (5, n, n) covariance forecasts
print(dcc.summary())
```

Model fit diagnostics are available on every model: `loglik_`, `aic`,
`bic`, `std_err`, `summary()`. See the class docs for options
(`mean`, `shrinkage`, `method`, `low_memory`, `n_jobs`).

Project home: https://github.com/mechanicpanic/multigarch
"""

from multigarch.models import CCC, DCC, GARCH

__all__ = ["GARCH", "CCC", "DCC"]
__version__ = "0.2.1"
