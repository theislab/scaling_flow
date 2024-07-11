from functools import partial

import jax
import numpy as np

__all__ = ["JaxSampler"]


class JaxSampler:
    """Data sampler for Jax."""

    def __init__(
        self,
        idcs_source: jax.Array,
        idcs_conditions: jax.Array,
        idcs_target: jax.Array,
        src_str_to_idx: dict[str, int],
        src_idx_to_data: dict[int, jax.Array],
        tgt_str_to_idx: dict[str, int],
        tgt_idx_to_data: dict[int, jax.Array],
        cond_str_to_idx: dict[str, int],
        cond_idx_to_data: dict[int, jax.Array],
        batch_size: int,
    ):
        assert len(idcs_source) == len(idcs_conditions)
        assert len(idcs_source) == len(idcs_target)
        """Initialize data sampler."""
        self.batch_size = batch_size
        self.idcs_source = idcs_source
        self.idcs_conditions = idcs_conditions
        self.idcs_target = idcs_target
        self.n_conditions = len(idcs_source)
        self.src_str_to_idx = src_str_to_idx
        self.src_idx_to_data = src_idx_to_data
        self.tgt_str_to_idx = tgt_str_to_idx
        self.tgt_idx_to_data = tgt_idx_to_data
        self.cond_str_to_idx = cond_str_to_idx
        self.cond_idx_to_data = cond_idx_to_data

        @partial(jax.jit, static_argnames=["dist_idx"])
        def _sample_batch(
            dist_idx: jax.Array,
            rng: jax.Array,
        ) -> dict[str, jax.Array]:
            """Jitted sample function."""
            rng_1, rng_2, rng_3 = jax.random.split(rng, 3)
            src_idx, conds_idx, tgt_idx = (
                self.idcs_source[dist_idx],
                self.idcs_conditions[dist_idx],
                self.idcs_target[dist_idx],
            )

            src = self.src_idx_to_data[src_idx]
            tgt = self.tgt_idx_to_data[tgt_idx]
            cond = np.tile(self.cond_idx_to_data[conds_idx], (len(src), 1))

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

            return {"src_lin": src[source_idcs], "tgt_lin": tgt[tgt_idcs], "src_condition": cond}

        self.sample_batch = _sample_batch
