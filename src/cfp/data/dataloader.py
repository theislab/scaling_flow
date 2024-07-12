from functools import partial

import jax
import numpy as np
from cfp.data.data import PerturbationData
__all__ = ["JaxSampler"]


class JaxSampler:
    """Data sampler for Jax."""

    def __init__(
        self,
        data: PerturbationData,
    ):
       self.perturbation_data = data


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
