from collections.abc import Callable, Iterable
from typing import Any

import jax
import numpy as np
from ott.neural.methods.flows import genot, otfm
from ott.solvers import utils as solver_utils
from tqdm import tqdm


class CellFlowTrainer:
    """Trainer for the OTFM/GENOT model with a conditional velocity field."""

    def __init__(
        self,
        dataloader: Iterable,
        model: otfm.OTFlowMatching | genot.GENOT,
    ):
        self.model = model
        self.dataloader = dataloader

    def train(
        self,
        num_iterations: int,
        valid_freq: int,
        callback_fn: Callable[[otfm.OTFlowMatching | genot.GENOT], Any],
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
        training_logs = {"loss": []}
        rng = jax.random.PRNGKey(0)

        pbar = tqdm(range(num_iterations))
        for it in pbar:
            rng, rng_resample, rng_step_fn = jax.random.split(rng, 3)
            idx = int(
                jax.random.randint(
                    rng, shape=[], minval=0, maxval=self.dataloader.n_conditions
                )
            )
            batch = self.dataloader.sample_batch(idx, rng)
            src, tgt = batch["src_lin"], batch["tgt_lin"]
            src_cond = batch.get("src_condition")

            if self.model.match_fn is not None:
                tmat = self.model.match_fn(src, tgt)
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

    def encode_conditions(self, condition: np.ndarray) -> np.ndarray:
        """Encode conditions

        Args:
            condition: Conditions to encode

        Returns
        -------
            Encoded conditions
        """
        return self.model.vf.apply(
            {"params": self.model.vf_state.params},
            condition,
            method="encode_conditions",
        )

    def predict(self, x: np.ndarray, condition: np.ndarray) -> np.ndarray:
        """Predict

        Args:
            x: Input data
            condition: Condition

        Returns
        -------
            Predicted output
        """
        return self.model.transport(x, condition)
