from typing import Any

import jax
import jax.numpy as jnp

from cfp.data.data import TrainingData

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

    def __init__(self, data: TrainingData, batch_size: int = 64):
        self.data = data
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
                rng_2,
                self.data.split_covariates_mask,
                [self.batch_size],
                replace=True,
                p=src_cond_p,
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
                self.data.perturbation_covariates_mask,
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
