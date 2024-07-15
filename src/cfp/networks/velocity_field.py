import functools
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from ott.neural.networks.layers import time_encoder

from cfp.networks.modules import MLPBlock
from cfp.networks import SetEncoder

__all__ = ["ConditionalVelocityField"]


class ConditionalVelocityField(nn.Module):
    """Parameterized neural vector field with conditions.

    Args:
        output_dim: Dimensionality of the output.
        condition_dim: Dimensionality of the condition.
        condition_encoder: Encoder for the condition.
        act_fn: Activation function.
        time_encoder: Encoder for the time.
        time_embedding_dims: Dimensions of the time embedding.
        time_dropout: Dropout rate for the time embedding.
        hidden_dims: Dimensions of the hidden layers.
        hidden_dropout: Dropout rate for the hidden layers.
        output_dims: Dimensions of the output layers.
        output_dropout: Dropout rate for the output layers.

    Returns
    -------
        Output of the neural vector field.
    """

    output_dim: int
    condition_dim: int = 0
    condition_encoder: Callable[[Any], jnp.ndarray] | None = None
    max_set_size: int = 2
    condition_encoder_kwargs: dict = dc_field(default_factory=dict)
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    time_embedding_dims: Sequence[int] = (1024, 1024, 1024)
    time_dropout: float = 0.0
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    hidden_dropout: float = 0.0
    output_dims: Sequence[int] = (1024, 1024, 1024)
    output_dropout: float = 0.0

    def setup(self):
        """Initialize the network."""
        if self.condition_encoder is not None:
            self.cond_encoder = SetEncoder(
                set_encoder=self.condition_encoder,
                max_set_size=self.max_set_size,
                act_fn=self.act_fn,
                **self.condition_encoder_kwargs,
            )
        self.time_encoder = MLPBlock(
            dims=self.time_embedding_dims,
            act_fn=self.act_fn,
            dropout_rate=self.time_dropout,
        )

        self.x_encoder = MLPBlock(
            dims=self.hidden_dims,
            act_fn=self.act_fn,
            dropout_rate=self.hidden_dropout,
        )

        self.decoder = MLPBlock(
            dims=self.output_dims,
            act_fn=self.act_fn,
            dropout_rate=self.output_dropout,
        )

        self.output_layer = nn.Dense(self.output_dim)

    def __call__(
        self,
        t: jnp.ndarray,
        x: jnp.ndarray,
        condition: jnp.ndarray,
        cond_sizes: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through the neural vector field.

        Args:
          t: Time of shape ``[batch, 1]``.
          x: Data of shape ``[batch, ...]``.
          condition: Conditioning vector of shape ``[batch, ...]``.
          training: If `True`, enables dropout for training.

        Returns
        -------
          Output of the neural vector field of shape ``[batch, output_dim]``.
        """
        if self.condition_encoder is not None:
            condition = self.cond_encoder(condition, cond_sizes, training)
        t = time_encoder.cyclical_time_encoder(t, n_freqs=1024)
        t = self.time_encoder(t, training)
        x = self.x_encoder(x, training)
        concatenated = jnp.concatenate((t, x, condition), axis=-1)
        out = self.decoder(concatenated, training)
        return self.output_layer(out)

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
    ) -> train_state.TrainState:
        """Create the training state.

        Args:
          rng: Random number generator.
          optimizer: Optimizer.
          input_dim: Dimensionality of the velocity field.

        Returns
        -------
          The training state.
        """
        t, x = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        cond = jnp.ones((1, self.max_set_size, self.condition_dim))
        cond_sizes = jnp.array([1])
        params = self.init(rng, t, x, cond, cond_sizes, training=False)["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply, params=params, tx=optimizer
        )
