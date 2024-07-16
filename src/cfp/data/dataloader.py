from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

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
        self.n_dists = np.sum(list(data.n_perturbations_given_control.values()))
        self.max_comb_size = self.data.max_comb_size

        def _sample_distributions(rng):
            rng_1, rng_2 = jax.random.split(rng)
            src_idx = jax.random.randint(
                rng_1,
                [
                    1,
                ],
                0,
                self.data.n_controls,
            )
            src_idx = src_idx.item()
            tgt_idx = jax.random.randint(
                rng_2,
                [
                    1,
                ],
                0,
                self.data.n_perturbations_given_control[src_idx],
            )
            tgt_idx = tgt_idx.item()
            return src_idx, tgt_idx

        def _sample(rng: jax.Array) -> Any:
            rng_1, rng_2 = jax.random.split(rng)
            src_idx, tgt_idx = _sample_distributions(rng_1)
            return _sample_batch(rng_2, src_idx, tgt_idx)

        @partial(jax.jit, static_argnames=["src_idx", "tgt_idx"])
        def _sample_batch(rng: jax.Array, src_idx: jax.Array, tgt_idx: jax.Array) -> dict[str, jax.Array]:
            """Jitted sample function."""
            rng_1, rng_2, rng_3 = jax.random.split(rng, 3)

            src = self.data.src_data[src_idx]
            tgt = self.data.tgt_data[src_idx][tgt_idx]
            conds_no_combination = [
                jnp.tile(self.data.tgt_data[src_idx][tgt_idx][pert_cond], (self.max_comb_size, 1))
                for pert_cond in self.data.perturbation_covariate_no_combination
            ]
            conds_combination = [
                self.data.tgt_data[src_idx][tgt_idx][pert_cond]
                for cond_group in self.data.perturbation_covariate_combinations
                for pert_cond in cond_group
            ]

            to_concat = []
            if len(conds_no_combination) > 0:
                conds_no_combination = jnp.hstack(conds_no_combination)
                to_concat.append(conds_no_combination)
            if len(conds_combination) > 0:
                conds_combination = jnp.vstack(conds_combination)
                to_concat.append(conds_combination)

            conds = jnp.concatenate(to_concat, axis=-1)
            print(conds.shape)
            conds = jnp.tile(conds, (self.batch_size, 1, 1))

            source_idcs = jax.random.choice(
                rng_2,
                len(src),
                replace=True,
                shape=[self.batch_size],
            )

            tgt_idcs = jax.random.choice(
                rng_3,
                len(tgt),
                replace=True,
                shape=[self.batch_size],
            )

            return {
                "src_lin": self.data.src_data[src_idx]["cell_data"][source_idcs],
                "tgt_lin": self.data.tgt_data[src_idx][tgt_idx]["cell_data"][tgt_idcs],
                "src_condition": conds,
            }

        self.sample = _sample
