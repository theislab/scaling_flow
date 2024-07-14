from importlib.metadata import version

from . import data, metrics, networks, training, model

__all__ = ["networks", "metrics", "data", "training", "model"]

__version__ = version("cell_flow_perturbation")
