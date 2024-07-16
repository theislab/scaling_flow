from typing import Any

import jax
import jax.numpy as jnp

from cfp.data.data import PerturbationData

__all__ = ["CFSampler"]


class CFSampler:
    """Data sampler for :class:`~cfp.data.data.PerturbationData`.

    Parameters
    ----------
    data : PerturbationData
        The data object to sample from.
    batch_size : int
        The batch size.
    """

    def __init__(self, data: PerturbationData, batch_size: int = 64):
        self.data = data
        self.batch_size = batch_size
        self.n_source_dists = data.n_controls
        self.n_target_dists = data.n_perturbed

        def _sample(rng: jax.Array) -> Any:
            rng_1, rng_2, rng_3, rng_4 = jax.random.split(rng, 4)
            source_dist_idx = jax.random.randint(rng_1, [1], 0, self.n_source_dists.item())
            source_cells = self.data.cell_data[self.data.split_covariates_mask == source_dist_idx]
            source_batch = jax.random.choice(rng_2, source_cells, (self.batch_size,), replace=True)
            target_dist_idx = jax.random.randint(
                rng_3, [1], 0, self.data.control_to_perturbation[source_dist_idx].shape[0]
            )
            target_cells = self.data.cell_data[self.data.perturbation_covariates_mask == target_dist_idx]
            target_batch = jax.random.choice(rng_4, target_cells, (self.batch_size,), replace=True)
            condition_batch = jnp.tile(self.data.conditions[self.condition_idx[target_dist_idx]], (self.batch_size, 1))

            return {
                "src_lin": source_batch,
                "tgt_lin": target_batch,
                "src_condition": condition_batch,
            }

        self.sample = _sample
