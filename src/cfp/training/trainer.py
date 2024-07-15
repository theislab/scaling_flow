from collections.abc import Callable, Iterable
from typing import Any

import jax
import numpy as np
from ott.neural.methods.flows import genot, otfm
from ott.solvers import utils as solver_utils
from tqdm import tqdm


class CellFlowTrainer:
    def __init__(
        self,
        dataloader: Iterable,
        model: otfm.OTFlowMatching | genot.GENOT,
    ):
        self.model = model
        self.dataloader = dataloader
        self.vector_field = ConditionalVelocityField()

    def train(
        self,
        num_iterations: int,
        valid_freq: int,
        callback_fn: Callable[[otfm.OTFlowMatching | genot.GENOT], Any],
    ) -> None:
        training_logs = {"loss": []}
        rng = jax.random.PRNGKey(0)
        for it in tqdm(range(num_iterations)):
            rng, rng_resample, rng_step_fn = jax.random.split(rng, 3)
            idx = int(jax.random.randint(rng, shape=[], minval=0, maxval=self.dataloader.n_conditions))
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

                callback_fn(
                    self.model,
                    log_metrics,
                )
