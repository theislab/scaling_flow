from importlib import metadata

from cfp import data, metrics, model, networks, solvers, training

__version__ = metadata.version("cell_flow_perturbation")
try:
    md = metadata.metadata(__name__)
    __version__ = md.get("version", "")
    __author__ = md.get("Author", "")
    __maintainer__ = md.get("Maintainer-email", "")
except ImportError:
    md = None
    __author__ = "TODO"

del metadata, md
