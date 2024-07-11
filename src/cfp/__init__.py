from importlib.metadata import version

from . import data, metrics, networks, training

__all__ = ["networks", "metrics", "data", "training"]

__version__ = version("cell_flow_perturbation")
