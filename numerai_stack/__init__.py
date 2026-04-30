"""Numerai Classic (v5.2) research + deployment stack.

Public sub-packages:
    data     - data access, caching, feature metadata
    cv       - purged walk-forward CV + per-era metrics
    models   - base learners (GBM, NN), neutralization, rank-gauss
    stack    - MMC-aware stacker + ensemble post-processing
    pipeline - training + inference orchestration
    compute  - Numerai Compute pickle upload / diagnostics
    tracking - experiment index
"""

__version__ = "0.1.0"

DEFAULT_DATA_VERSION = "v5.2"
