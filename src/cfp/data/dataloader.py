from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from cfp.data.data import TrainingData, ValidationData

__all__ = ["TrainSampler"]


class TrainSampler:
    """Data sampler for :class:`~cfp.data.data.TrainingData`.

    Parameters
    ----------
    data : TrainingData
        The data object to sample from.
    batch_size : int
        The batch size.
    """

    def __init__(self, data: TrainingData, batch_size: int = 1024):
        self.data = data
        self.data_idcs = jnp.arange(data.cell_data.shape[0])
        self.batch_size = batch_size
        self.n_source_dists = data.n_controls
        self.n_target_dists = data.n_perturbations
        self.conditional_samplings = [
            lambda key: jax.random.choice(
                key, self.data.control_to_perturbation[i].shape[0]  # noqa: B023
            )
            for i in range(self.n_source_dists)
        ]
        self.get_embeddings = lambda idx: {
            pert_cov: jnp.expand_dims(arr[idx], 0)
            for pert_cov, arr in self.data.condition_data.items()
        }

        @jax.jit
        def _sample(rng: jax.Array) -> Any:
            rng_1, rng_2, rng_3, rng_4 = jax.random.split(rng, 4)
            source_dist_idx = jax.random.choice(rng_1, self.n_source_dists)
            source_cells_mask = self.data.split_covariates_mask == source_dist_idx
            src_cond_p = source_cells_mask / jnp.count_nonzero(source_cells_mask)
            source_batch_idcs = jax.random.choice(
                rng_2, self.data_idcs, [self.batch_size], replace=True, p=src_cond_p
            )

            source_batch = self.data.cell_data[source_batch_idcs]

            target_dist_idx = jax.lax.switch(
                source_dist_idx, self.conditional_samplings, rng_3
            )
            target_cells_mask = (
                self.data.perturbation_covariates_mask == target_dist_idx
            )
            tgt_cond_p = target_cells_mask / jnp.count_nonzero(target_cells_mask)
            target_batch_idcs = jax.random.choice(
                rng_4,
                self.data_idcs,
                [self.batch_size],
                replace=True,
                p=tgt_cond_p,
            )
            target_batch = self.data.cell_data[target_batch_idcs]
            if self.data.condition_data is None:
                return {"src_cell_data": source_batch, "tgt_cell_data": target_batch}

            condition_batch = self.get_embeddings(target_dist_idx)
            return {
                "src_cell_data": source_batch,
                "tgt_cell_data": target_batch,
                "condition": condition_batch,
            }

        self.sample = _sample


class ValidationSampler:
    def __init__(self, val_data: ValidationData, seed: int = 0) -> None:
        self.val_data = val_data
        self.perturbation_to_control = self._get_perturbation_to_control(val_data)
        self.n_conditions_on_log_iteration = (
            val_data.n_conditions_on_log_iteration
            if val_data.n_conditions_on_log_iteration is not None
            else val_data.n_perturbations
        )
        self.n_conditions_on_train_end = (
            val_data.n_conditions_on_train_end
            if val_data.n_conditions_on_train_end is not None
            else val_data.n_perturbations
        )
        self.rng = np.random.default_rng(seed)
        if self.val_data.condition_data is None:
            raise NotImplementedError("Validation data must have condition data.")

    def sample(self, mode: Literal["on_log_iteration", "on_train_end"]) -> Any:
        size = (
            self.n_conditions_on_log_iteration
            if mode == "on_log_iteration"
            else self.n_conditions_on_train_end
        )
        condition_idcs = self.rng.choice(
            self.val_data.n_perturbations, size=(size,), replace=False
        )

        source_idcs = [
            self.perturbation_to_control[cond_idx] for cond_idx in condition_idcs
        ]
        source_cells_mask = [
            self.val_data.split_covariates_mask == source_idx
            for source_idx in source_idcs
        ]
        source_cells = [self.val_data.cell_data[mask] for mask in source_cells_mask]
        target_cells_mask = [
            cond_idx == self.val_data.perturbation_covariates_mask
            for cond_idx in condition_idcs
        ]
        target_cells = [self.val_data.cell_data[mask] for mask in target_cells_mask]
        conditions = [self._get_condition_data(cond_idx) for cond_idx in condition_idcs]
        cell_rep_dict = {}
        cond_dict = {}
        true_dict = {}
        for i in range(len(condition_idcs)):
            arr = self.val_data.perturbation_idx_to_covariates[condition_idcs[i]]
            k = tuple(arr[i] for i in range(len(arr)))
            cell_rep_dict[k] = source_cells[i]
            cond_dict[k] = conditions[i]
            true_dict[k] = target_cells[i]

        return {"source": cell_rep_dict, "condition": cond_dict, "target": true_dict}

    def _get_perturbation_to_control(self, val_data: ValidationData) -> dict[int, int]:
        d = {}
        for k, v in val_data.control_to_perturbation.items():
            for el in v:
                d[el] = k
        return d

    def _get_condition_data(self, cond_idx: int) -> jnp.ndarray:
        return {k: v[[cond_idx], ...] for k, v in self.val_data.condition_data.items()}
