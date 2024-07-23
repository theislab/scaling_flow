from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
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
    ):
        if not isinstance(model, (otfm.OTFlowMatching | genot.GENOT)):
            raise NotImplementedError(
                f"Model must be an instance of OTFlowMatching or GENOT, got {type(model)}"
            )

        self.model = model

    def _genot_step_fn(self, rng, src, tgt, condition):
        """Step function for GENOT solver."""
        rng, rng_resample, rng_noise, rng_time, rng_step_fn = jax.random.split(rng, 5)

        matching_data = (src, tgt)
        n = src.shape[0]

        time = self.model.time_sampler(rng_time, n * self.model.n_samples_per_src)
        latent = self.model.latent_noise_fn(
            rng_noise, (n, self.model.n_samples_per_src)
        )

        tmat = self.model.data_match_fn(*matching_data)  # (n, m)

        src_ixs, tgt_ixs = solver_utils.sample_conditional(  # (n, k), (m, k)
            rng_resample,
            tmat,
            k=self.model.n_samples_per_src,
        )

        src, tgt = src[src_ixs], tgt[tgt_ixs]  # (n, k, ...),  # (m, k, ...)
        if condition is not None:
            condition = jnp.tile(condition, (src.shape[0], 1, 1))

        if self.model.latent_match_fn is not None:
            src, condition, tgt = self.model._match_latent(
                rng, src, condition, latent, tgt
            )

        src = src.reshape(-1, *src.shape[2:])  # (n * k, ...)
        tgt = tgt.reshape(-1, *tgt.shape[2:])  # (m * k, ...)
        latent = latent.reshape(-1, *latent.shape[2:])
        if condition is not None:
            condition = condition.reshape(-1, *condition.shape[2:])

        loss, self.model.vf_state = self.model.step_fn(
            rng_step_fn, self.model.vf_state, time, src, tgt, latent, condition
        )
        return loss

    def _otfm_step_fn(self, rng, src, tgt, condition):
        """Step function for OTFM solver."""
        rng_resample, rng_step_fn = jax.random.split(rng, 2)
        if self.model.match_fn is not None:
            tmat = self.model.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]

        self.model.vf_state, loss = self.model.step_fn(
            rng_step_fn,
            self.model.vf_state,
            src,
            tgt,
            condition,
        )
        return loss

    def train(
        self,
        dataloader: CFSampler,
        val_data: dict[str, ArrayLike],
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
            rng, rng_step_fn = jax.random.split(rng, 2)
            batch = dataloader.sample(rng)

            src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
            condition = batch.get("condition")

            if isinstance(self.model, genot.GENOT):
                loss = self._genot_step_fn(rng_step_fn, src, tgt, condition)
            else:
                loss = self._otfm_step_fn(rng_step_fn, src, tgt, condition)

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
            method="get_condition_embedding",
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

        def vf(
            t: jnp.ndarray, x: jnp.ndarray, cond: dict[str, jnp.ndarray] | None
        ) -> jnp.ndarray:
            params = self.model.vf_state.params
            return self.model.vf_state.apply_fn(
                {"params": params}, t, x, cond, train=False
            )

        def solve_ode(x: jnp.ndarray, cond: jnp.ndarray | None) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
                args=cond,
                dt0=None,
                solver=diffrax.Tsit5(),
                stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
            )
            return result.ys[0]

        x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, None]))(x, condition)
        return np.array(x_pred)
