from importlib.metadata import version

from . import data, metrics, model, networks, training

__all__ = ["networks", "metrics", "data", "training", "model"]

__version__ = version("cell_flow_perturbation")
