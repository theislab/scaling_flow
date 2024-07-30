from collections.abc import Callable, Sequence
from typing import Any, Literal

import jax
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from cfp.data.data import ValidationData
from cfp.data.dataloader import TrainSampler
from cfp.solvers import genot, otfm
from cfp.training.callbacks import CallbackRunner


class CellFlowTrainer:
    """Trainer for the OTFM/GENOT model with a conditional velocity field.

    Args:
        dataloader: Data sampler.
        model: OTFM/GENOT model with a conditional velocity field.
        seed: Random seed for subsampling validation data.

    Returns
    -------
        None
    """

    def __init__(
        self,
        model: otfm.OTFlowMatching | genot.GENOT,
        seed: int = 0,
    ):
        if not isinstance(model, (otfm.OTFlowMatching | genot.GENOT)):
            raise NotImplementedError(
                f"Model must be an instance of OTFlowMatching or GENOT, got {type(model)}"
            )

        self.model = model
        self.rng_subsampling = np.random.default_rng(seed)
        self.training_logs: dict[str, Any] = {}

    def _sample_validation_data(
        self,
        valid_data: dict[str, ValidationData],
        stage: Literal["on_log_iteration", "on_train_end"],
    ) -> dict[str, ValidationData]:
        """Sample validation data for computing metrics"""
        if stage == "on_train_end":
            n_conditions_to_sample = lambda x: x.n_conditions_on_train_end
        elif stage == "on_log_iteration":
            n_conditions_to_sample = lambda x: x.n_conditions_on_log_iteration
        else:
            raise ValueError(f"Stage {stage} not supported.")
        subsampled_validation_data = {}
        for val_data_name, val_data in valid_data.items():
            if n_conditions_to_sample(val_data) == -1:
                subsampled_validation_data[val_data_name] = val_data
            else:
                condition_idxs = self.rng_subsampling.choice(
                    len(val_data.condition_data),
                    n_conditions_to_sample(val_data),
                    replace=False,
                )
                subsampled_validation_data[val_data_name] = (
                    self._extract_subsampled_validation_data(
                        val_data,
                        condition_idxs,
                    )
                )

        return subsampled_validation_data

    def _extract_subsampled_validation_data(
        self,
        val_data: ValidationData,
        condition_idxs: ArrayLike,
    ) -> ValidationData:
        """Extract subsampled validation data."""
        src_data = {}
        tgt_data = {}
        condition_data = {}
        for cond_idx in condition_idxs:
            for src_idx in val_data.src_data.keys():
                src_data[src_idx] = val_data.src_data[src_idx]

                tgt_data[src_idx] = {}
                if cond_idx in val_data.tgt_data[src_idx]:
                    print("src_idx", src_idx)
                    print("idx", cond_idx)
                    tgt_data[src_idx][cond_idx] = val_data.tgt_data[src_idx][cond_idx]
                    condition_data[cond_idx] = val_data.condition_data[cond_idx]

        return ValidationData(
            src_data=src_data,
            tgt_data=tgt_data,
            condition_data=condition_data,
            max_combination_length=val_data.max_combination_length,
        )

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
                    pred = self.model.predict(src, condition)
                    valid_pred_data[val_key][src_dist][tgt_dist] = pred

        src = batch["src_cell_data"]
        condition = batch.get("condition")
        train_pred = self.model.predict(src, condition)
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
        )
        crun.on_train_begin()

        pbar = tqdm(range(num_iterations))
        for it in pbar:
            rng, rng_step_fn = jax.random.split(rng, 2)
            batch = dataloader.sample(rng)
            loss = self.model.step_fn(rng_step_fn, batch)
            self.training_logs["loss"].append(float(loss))

            if ((it - 1) % valid_freq == 0) and (it > 1):
                # TODO: Accumulate tran data over multiple iterations

                # Subsample validation data
                valid_data_subsampled = self._sample_validation_data(
                    valid_data, stage="on_log_iteration"
                )

                # Get predictions from validation data
                train_data, valid_pred_data = self._validation_step(
                    batch, valid_data_subsampled
                )

                # Run callbacks
                metrics = crun.on_log_iteration(
                    valid_data_subsampled, train_data, valid_pred_data
                )
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
            valid_data_subsampled = self._sample_validation_data(
                valid_data, stage="on_train_end"
            )

            # Val step and callbacks at the end of training
            train_data, valid_pred_data = self._validation_step(
                batch, valid_data_subsampled
            )
            metrics = crun.on_train_end(
                valid_data_subsampled, train_data, valid_pred_data
            )
            self._update_logs(metrics)
