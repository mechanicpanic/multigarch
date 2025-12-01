# multigarch

Simple Multivariate GARCH models for Python 3.10+.

Features:
- JIT-compiled inner loops (numba)
- Parallel GARCH fitting (joblib)
- Configurable GARCH(p,q) orders

## Installation

```bash
pip install multigarch
```

## Usage

```python
import numpy as np
from multigarch import GARCH, CCC, DCC

# Generate sample returns (T periods, n assets)
returns = np.random.randn(500, 30) * 0.02

# Univariate GARCH(1,1)
garch = GARCH().fit(returns[:, 0])
garch.forecast(horizon=5)

# CCC: Constant Conditional Correlation (fastest)
ccc = CCC().fit(returns)

# DCC: Dynamic Conditional Correlation
dcc = DCC().fit(returns)
dcc.forecast(horizon=5)  # shape: (5, 30, 30)

# Access results
dcc.H  # time-varying covariances (T, n, n)
dcc.R  # time-varying correlations (T, n, n)
ccc.R  # constant correlation matrix (n, n)

# Configure GARCH orders and parallelism
dcc = DCC(p=2, q=1, n_jobs=4).fit(returns)
```

## Models

| Model | Description | Speed |
|-------|-------------|-------|
| `GARCH(p,q)` | Univariate GARCH | - |
| `CCC(p,q)` | Constant Conditional Correlation | Fast |
| `DCC(p,q)` | Dynamic Conditional Correlation | Fast (JIT) |

## Parameters

- `p`: ARCH order (default 1)
- `q`: GARCH order (default 1)
- `n_jobs`: Parallel jobs for fitting, -1 = all cores (CCC/DCC only)

## License

MIT
