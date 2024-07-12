import numpy as np
import anndata as ad

from ott.neural.methods.flows import otfm, genot
from cfp.training.trainer import CellFlowTrainer


class CellFlow:
    def __init__(
        self, adata: ad.AnnData, solver: str, condition_encoder: str, **kwargs
    ):
        self.adata = adata
        self.solver = solver
        self.condition_encoder = condition_encoder
        self.model = None

        self.dataloader = _get_dataloader(self.adata, **kwargs)

    def train(self, **kwargs):
        if solver == "otfm":
            solver = otfm.OTFlowmatching
        elif solver == "genot":
            solver = genot.GENOT

        self.trainer = CellFlowTrainer(
            model=solver,
            dl=self.dl,
            set_encoder=self.condition_encoder,
        )

        self.model = self.trainer.train(model, self.adata, **kwargs)
