import numpy as np

# TODO(michalk8): polish

try:
    from numpy.typing import NDArray

    ArrayLike = NDArray[np.float64]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]
    DTypeLike = np.dtype  # type: ignore[misc]
