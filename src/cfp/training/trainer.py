from collections.abc import Callable, Sequence
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from ott.neural.methods.flows import genot, otfm
from ott.solvers import utils as solver_utils
from tqdm import tqdm

from cfp.data.dataloader import TrainSampler
from cfp.training.callbacks import CallbackRunner


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
        self.training_logs: dict[str, Any] = {}

        @jax.jit
        def _otfm_step_fn(
            rng: jnp.ndarray,
            src: jnp.ndarray,
            tgt: jnp.ndarray,
            condition: jnp.ndarray,
        ):
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

        self._otfm_step_fn = _otfm_step_fn

    def _genot_step_fn(
        self,
        rng: jnp.ndarray,
        src: jnp.ndarray,
        tgt: jnp.ndarray,
        condition: jnp.ndarray,
    ):
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

    def _validation_step(
        self,
        batch: dict[str, ArrayLike],
        val_data: dict[str, dict[str, dict[str, ArrayLike]]],
    ) -> dict[str, dict[str, dict[str, ArrayLike]]]:
        """Compute predictions for validation data."""
        # TODO: Sample fixed number of conditions to validate on

        valid_pred_data: dict[str, dict[str, ArrayLike]] = {}
        for val_key, vdl in val_data.items():
            valid_pred_data[val_key] = {}
            for src_dist in vdl.src_data:
                valid_pred_data[val_key][src_dist] = {}
                src = vdl.src_data[src_dist]
                tgt_dists = vdl.tgt_data[src_dist]
                for tgt_dist in tgt_dists:
                    condition = vdl.condition_data[src_dist][tgt_dist]
                    pred = self.predict(src, condition)
                    valid_pred_data[val_key][src_dist][tgt_dist] = pred

        src = batch["src_cell_data"]
        condition = batch.get("condition")
        train_pred = self.predict(src, condition)
        batch["pred_data"] = train_pred

        return batch, valid_pred_data

    def _update_logs(self, logs: dict[str, Any]) -> None:
        """Update training logs."""
        for k, v in logs.items():
            if k not in self.training_logs:
                self.training_logs[k] = []
            self.training_logs[k].append(v)

    def train(
        self,
        dataloader: TrainSampler,
        num_iterations: int,
        valid_freq: int,
        valid_data: dict[str, dict[str, dict[str, ArrayLike]]] | None = None,
        monitor_metrics: Sequence[str] = [],
        callbacks: Sequence[Callable] = [],
    ) -> None:
        """Trains the model.

        Args:
            num_iterations: Number of iterations to train the model.
            batch_size: Batch size.
            valid_freq: Frequency of validation.
            callbacks: Callback functions.
            monitor_metrics: Metrics to monitor.

        Returns
        -------
            None
        """
        self.training_logs = {"loss": []}
        rng = jax.random.PRNGKey(0)

        # Initiate callbacks
        valid_data = valid_data or {}
        crun = CallbackRunner(
            callbacks=callbacks,
            data=valid_data,
        )
        crun.on_train_begin()

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

            self.training_logs["loss"].append(float(loss))

            if ((it - 1) % valid_freq == 0) and (it > 1):
                # TODO: Accumulate tran data over multiple iterations

                # Get predictions from validation data
                train_data, valid_pred_data = self._validation_step(batch, valid_data)

                # Run callbacks
                metrics = crun.on_log_iteration(train_data, valid_pred_data)
                self._update_logs(metrics)

                # Update progress bar
                mean_loss = np.mean(self.training_logs["loss"][-valid_freq:])
                postfix_dict = {
                    metric: round(self.training_logs[metric][-1], 3)
                    for metric in monitor_metrics
                }
                postfix_dict["loss"] = round(mean_loss, 3)
                pbar.set_postfix(postfix_dict)

        if num_iterations > 0:
            # Val step and callbacks at the end of training
            train_data, valid_pred_data = self._validation_step(batch, valid_data)
            metrics = crun.on_train_end(train_data, valid_pred_data)
            self._update_logs(metrics)

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
