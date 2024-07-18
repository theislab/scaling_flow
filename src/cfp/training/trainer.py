from collections.abc import Callable
from typing import Any

import jax
import numpy as np
from numpy.typing import ArrayLike
from ott.neural.methods.flows import genot, otfm
from ott.solvers import utils as solver_utils
from tqdm import tqdm

from cfp.data.dataloader import CFSampler


class CellFlowTrainer:
    """Trainer for the OTFM/GENOT model with a conditional velocity field.

    Args:
        dataloader: Data sampler.
        model: OTFM/GENOT model with a conditional velocity field.


    Returns
    -------
        None
    """

    def __init__(
        self,
        model: otfm.OTFlowMatching | genot.GENOT,
        match_fn: Callable[[ArrayLike, ArrayLike], Any] | None = None,
    ):
        self.model = model
        self.match_fn = match_fn

    def train(
        self,
        dataloader: CFSampler,
        num_iterations: int,
        valid_freq: int,
        callback_fn: (
            Callable[[otfm.OTFlowMatching | genot.GENOT, dict[str, Any]], Any] | None
        ) = None,
    ) -> None:
        """Trains the model.

        Args:
            num_iterations: Number of iterations to train the model.
            valid_freq: Frequency of validation.
            callback_fn: Callback

        Returns
        -------
            None
        """
        training_logs: dict[str, Any] = {"loss": []}
        rng = jax.random.PRNGKey(0)

        pbar = tqdm(range(num_iterations))
        for it in pbar:
            rng, rng_resample, rng_step_fn = jax.random.split(rng, 3)
            batch = dataloader.sample(rng)

            src, tgt = batch["src_lin"], batch["tgt_lin"]
            src_cond = batch.get("src_condition")

            if self.match_fn is not None:
                tmat = self.match_fn(src, tgt)
                src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
                src, tgt = src[src_ixs], tgt[tgt_ixs]
                src_cond = None if src_cond is None else src_cond[src_ixs]

            self.model.vf_state, loss = self.model.step_fn(
                rng_step_fn,
                self.model.vf_state,
                src,
                tgt,
                src_cond,
            )

            training_logs["loss"].append(float(loss))
            if ((it - 1) % valid_freq == 0) and (it > 1):
                train_loss = np.mean(training_logs["loss"][valid_freq:])
                log_metrics = {"train_loss": train_loss}

                pbar.set_postfix({"loss": float(loss.mean().round(2))})

                if callback_fn is not None:
                    callback_fn(
                        self.model,
                        log_metrics,
                    )

    def get_condition_embedding(self, condition: ArrayLike) -> ArrayLike:
        """Encode conditions

        Args:
            condition: Conditions to encode

        Returns
        -------
            Encoded conditions
        """
        cond_embed = self.model.vf.apply(
            {"params": self.model.vf_state.params},
            condition,
            method="encode_conditions",
        )
        return np.array(cond_embed)

    def predict(self, x: ArrayLike, condition: ArrayLike) -> ArrayLike:
        """Predict

        Args:
            x: Input data
            condition: Condition

        Returns
        -------
            Predicted output
        """
        x_pred = self.model.transport(x, condition)
        return np.array(x_pred)
