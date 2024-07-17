from typing import Literal, Any, Sequence, Callable
from dataclasses import field as dc_field

import optax
import anndata as ad
from numpy.typing import ArrayLike
from ott.neural.methods.flows import genot, otfm

from cfp.training.trainer import CellFlowTrainer
from cfp.networks.velocity_field import ConditionalVelocityField
from cfp.data.dataloader import CFSampler


class CellFlow:
    """CellFlow model for perturbation preduction using FlowMatching.

    Args:
        adata: Anndata object.
        solver: Solver to use for training the model.

    """

    def __init__(self, adata: ad.AnnData, solver: Literal["otfm", "genot"] = "otfm"):
        self.adata = adata
        self.dataloader = None
        self.solver = solver
        self.model = None
        self._solver = None

    def prepare_data(
        # TODO: Adapt to new PerturbationData
        self,
        condition_dim: int = 3,
        max_set_size: int = 2,
        **kwargs,
    ) -> None:
        """Prepare dataloader for training from anndata object.

        **kwargs: Keyword arguments for CFSampler.
        """
        # TODO: Adapt to new PerturbationData
        # Should also set attributes for the model such as
        # * max_set_size
        # * condition_dim
        self._max_set_size = max_set_size
        self._condition_dim = condition_dim
        self.dataloader = CFSampler(**kwargs)

    def prepare_model(
        self,
        condition_encoder: Literal["transformer", "deepset"] = "transformer",
        condition_embedding_dim: int = 32,
        condition_encoder_kwargs: dict[str, Any] = dc_field(default_factory=dict),
        time_encoder_dims: Sequence[int] = (1024, 1024, 1024),
        time_encoder_dropout: float = 0.0,
        hidden_dims: Sequence[int] = (1024, 1024, 1024),
        hidden_dropout: float = 0.0,
        decoder_dims: Sequence[int] = (1024, 1024, 1024),
        decoder_dropout: float = 0.0,
        optimizer: optax.GradientTransformation = optax.adam(1e-3),
        seed=0,
    ) -> None:
        """Prepare model for training.

        Args:
            condition_encoder: Encoder for the condition.
            condition_embedding_dim: Dimensions of the condition embedding.
            condition_encoder_kwargs: Keyword arguments for the condition encoder.
            time_encoder_dims: Dimensions of the time embedding.
            time_encoder_dropout: Dropout rate for the time embedding.
            hidden_dims: Dimensions of the hidden layers.
            hidden_dropout: Dropout rate for the hidden layers.
            decoder_dims: Dimensions of the output layers.
            decoder_dropout: Dropout rate for the output layers.
            optimizer: Optimizer for training.
            seed: Random seed.

        Returns
        -------
            None
        """
        if self.dataloader is None:
            raise ValueError(
                "Dataloader not initialized. Please call prepare_data first."
            )

        vf = ConditionalVelocityField(
            output_dim=self.adata.shape[1],
            condition_encoder=condition_encoder,
            condition_dim=self._condition_dim,
            condition_embedding_dim=condition_embedding_dim,
            max_set_size=self._max_set_size,
            condition_encoder_kwargs=condition_encoder_kwargs,
            time_encoder_dims=time_encoder_dims,
            time_encoder_dropout=time_encoder_dropout,
            hidden_dims=hidden_dims,
            hidden_dropout=hidden_dropout,
            decoder_dims=decoder_dims,
            decoder_dropout=decoder_dropout,
        )

        if self.solver == "otfm":
            self._solver = otfm.OTFlowMatching(
                vf=vf,
                match_fn=genot.match_linear,
                flow=genot.ConstantNoiseFlow(0.0),
                optimizer=optimizer,
                rng=jax.random.PRNGKey(seed),
            )
        elif self.solver == "genot":
            self._solver = genot.GENOT(
                vf=vf,
                optimizer=optimizer,
                rng=jax.random.PRNGKey(seed),
            )
        # NOTE: The use of "model" is a bit confusing here, maybe we can harmonize the
        # naming a bit better
        self.model = CellFlowTrainer(dataloader=self.dataloader, model=self._solver)

    def train(
        self,
        num_iterations: int,
        valid_freq: int,
        callback_fn: (
            Callable[[otfm.OTFlowMatching | genot.GENOT, dict[str, Any]], Any] | None
        ) = None,
    ) -> None:
        """Train the model.

        Args:
            num_iterations: Number of iterations to train the model.
            valid_freq: Frequency of validation.
            callback_fn: Callback function.

        Returns
        -------
            None
        """
        if self.model is None:
            raise ValueError("Model not initialized. Please call prepare_model first.")

        self.model.train(
            num_iterations=num_iterations,
            valid_freq=valid_freq,
            callback_fn=callback_fn,
        )

    def predict(
        self,
        adata: ad.AnnData,
    ) -> ArrayLike:
        """Predict perturbation.

        Args:
            adata: Anndata object.


        Returns
        -------
            Perturbation prediction.
        """
        pass

    def get_condition_embedding(
        self,
        adata: ad.AnnData,
    ) -> ArrayLike:
        """Get condition embedding.

        Args:
            adata: Anndata object.

        Returns
        -------
            Condition embedding.
        """
        pass
