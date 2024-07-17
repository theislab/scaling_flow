import anndata as ad
from ott.neural.methods.flows import genot, otfm

from cfp.training.trainer import CellFlowTrainer


class CellFlow:
    """CellFlow model for perturbation preduction using FlowMatching."""

    def __init__(
        self, adata: ad.AnnData, solver: str, condition_encoder: str, **kwargs
    ):
        self.adata = adata
        self.solver = solver
        self.condition_encoder = condition_encoder
        self.model = None

        self.dataloader = self._get_dataloader(self.adata, **kwargs)

    def train(self, **kwargs):
        """Train the model."""
        if self.solver == "otfm":
            solver = otfm.OTFlowmatching
        elif self.solver == "genot":
            solver = genot.GENOT

        self.trainer = CellFlowTrainer(
            model=solver,
            dl=self.dl,
            set_encoder=self.condition_encoder,
        )

        self.model = self.trainer.train(**kwargs)

    def _get_dataloader(self, adata, **kwargs):
        """Get dataloader for the model."""
        return None
