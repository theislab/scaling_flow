from importlib import metadata

from cellflow import data, datasets, metrics, model, networks, solvers, training, utils

try:
    from cellflow import preprocessing
except ImportError as e:
    raise ImportError(
        "The 'preprocessing' module is not installed. If required, please install 'cellflow[preprocessing]'"
    ) from e

try:
    from cellflow import external
except ImportError as e:
    raise ImportError("The 'external' module is not installed. If required, please install 'cellflow[external]'") from e
