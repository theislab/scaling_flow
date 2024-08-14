from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from flax import linen as nn


# TODO: this is also defined in _set_encoders
class MLPBlock(nn.Module):
    """An MLP block."""

    dims: Sequence[int] = (1024, 1024, 1024)
    act_fn: Any = nn.silu
    dropout_rate: float = 0.0
    training: bool | None = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool | None = None,
    ) -> jnp.ndarray:
        """Apply the MLP block.

        Args:
          x: Input data of shape (batch_size, dims[0]).

        Returns
        -------
          Output data of shape (batch_size, dims[-1]).
        """
        training = nn.merge_param("training", self.training, training)
        is_eval = not training

        for dim in self.dims:
            x = nn.Dense(dim)(x)
            x = self.act_fn(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=is_eval)
        return x
