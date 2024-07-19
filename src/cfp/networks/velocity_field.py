import warnings
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from ott.neural.networks.layers import time_encoder

from cfp import logger
from cfp.networks.modules import MLPBlock
from cfp.networks.set_encoders import ConditionEncoder

__all__ = ["ConditionalVelocityField"]


class ConditionalVelocityField(nn.Module):
    """Parameterized neural vector field with conditions.

    Args:
        output_dim: Dimensionality of the output.
        condition_dim: Dimensionality of the condition.
        condition_encoder: Encoder for the condition.
        condition_embedding_dim: Dimensions of the condition embedding.
        max_set_size: Maximum size of the set.
        condition_encoder_kwargs: Keyword arguments for the condition encoder.
        act_fn: Activation function.
        time_freqs: Frequency of the cyclical time encoding.
        time_encoder_dims: Dimensions of the time embedding.
        time_encoder_dropout: Dropout rate for the time embedding.
        hidden_dims: Dimensions of the hidden layers.
        hidden_dropout: Dropout rate for the hidden layers.
        decoder_dims: Dimensions of the output layers.
        decoder_dropout: Dropout rate for the output layers.

    Returns
    -------
        Output of the neural vector field.
    """

    output_dim: int
    max_combination_length: int
    encode_conditions: bool = True
    condition_dim: int = 0
    condition_embedding_dim: int = 32
    condition_encoder_kwargs: dict[str, Any] = dc_field(default_factory=dict)
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    time_freqs: int = 1024
    time_encoder_dims: Sequence[int] = (1024, 1024, 1024)
    time_encoder_dropout: float = 0.0
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    hidden_dropout: float = 0.0
    decoder_dims: Sequence[int] = (1024, 1024, 1024)
    decoder_dropout: float = 0.0

    def setup(self):
        """Initialize the network."""
        if self.encode_conditions:
            self.condition_encoder = ConditionEncoder(
                output_dim=self.condition_embedding_dim,
                **self.condition_encoder_kwargs,
            )
        self.time_encoder = MLPBlock(
            dims=self.time_encoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.time_encoder_dropout,
        )

        self.x_encoder = MLPBlock(
            dims=self.hidden_dims,
            act_fn=self.act_fn,
            dropout_rate=self.hidden_dropout,
        )

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
        )

        self.output_layer = nn.Dense(self.output_dim)

    def __call__(
        self,
        t: jnp.ndarray,
        x: jnp.ndarray,
        condition: dict[str, jnp.ndarray],
        train: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through the neural vector field.

        Args:
            t: Time of shape ``[batch, 1]``.
            x: Data of shape ``[batch, ...]``.
            condition: Condition dictionary, with condition names as keys and condition representations of shape ``[batch, max_combination_length, condition_dim]`` as values.
            train: If `True`, enables dropout for training.

        Returns
        -------
            Output of the neural vector field of shape ``[batch, output_dim]``.
        """
        if self.encode_conditions:
            condition = self.condition_encoder(condition, training=train)
        else:
            condition = jnp.concatenate(list(condition.values()), axis=-1)
        t = time_encoder.cyclical_time_encoder(t, n_freqs=1024)
        t = self.time_encoder(t, training=train)
        x = self.x_encoder(x, training=train)
        concatenated = jnp.concatenate((t, x, condition), axis=-1)
        out = self.decoder(concatenated, training=train)
        return self.output_layer(out)

    def get_condition_embedding(self, condition: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Get the embedding of the condition.

        Args:
            condition: Conditioning vector of shape ``[batch, ...]``.

        Returns
        -------
            Embedding of the condition.
        """
        if self.encode_conditions:
            condition = self.condition_encoder(condition, training=False)
        else:
            condition = jnp.concatenate(list(condition.values()), axis=-1)
            logger.warning(
                "Condition encoder is not defined. Returning concatenated input as the embedding."
            )
        return condition

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        condition_dim: int | None = None,
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
        condition_dim = condition_dim or self.condition_dim
        t, x = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        cond = jnp.ones((1, self.max_combination_length, condition_dim))
        params = self.init(rng, t, x, cond, train=False)["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply, params=params, tx=optimizer
        )

    @property
    def output_dims(self):
        """Dimonsions of the output layers."""
        return tuple(self.decoder_dims) + (self.output_dim,)
