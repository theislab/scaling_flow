from importlib.metadata import version

from . import data, metrics, networks, training, models

__all__ = ["networks", "metrics", "data", "training", "models"]

__version__ = version("cell_flow_perturbation")
