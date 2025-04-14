from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from ott.neural.networks.layers import time_encoder

from cellflow._logging import logger
from cellflow._types import Layers_separate_input_t, Layers_t
from cellflow.networks._set_encoders import ConditionEncoder
try:
    from cellflow.networks._prophet_adapter import ProphetEncoder
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False
from cellflow.networks._utils import MLPBlock

__all__ = ["ConditionalVelocityField", "GENOTConditionalVelocityField"]


class ConditionalVelocityField(nn.Module):
    """Parameterized neural vector field with conditions.

    Parameters
    ----------
        output_dim
            Dimensionality of the output.
        max_combination_length
            Maximum number of covariates in a combination.
        condition_mode
            Mode of the encoder, should be one of:
            - ``'deterministic'``: Learns condition encoding point-wise.
            - ``'stochastic'``: Learns a Gaussian distribution for representing conditions.
            - ``'prophet'``: Uses Prophet's transformer-based encoder architecture.
        regularization
            Regularization strength in the latent space:
            - For deterministic mode, it is the strength of the L2 regularization.
            - For stochastic mode, it is the strength of the KL divergence regularization.
        encode_conditions
                Processes the embedding of the perturbation conditions if :obj:`True`. If
                :obj:`False`, directly inputs the embedding of the perturbation conditions to the
                generative velocity field. In the latter case, ``'condition_embedding_dim'``,
                ``'condition_encoder_kwargs'``, ``'pooling'``, ``'pooling_kwargs'``,
                ``'layers_before_pool'``, ``'layers_after_pool'``, ``'cond_output_dropout'``
                are ignored.
        condition_embedding_dim
            Dimensions of the condition embedding.
        covariates_not_pooled
            Covariates that will escape pooling (should be identical across all set elements).
        pooling
            Pooling method.
        pooling_kwargs
            Keyword arguments for the pooling method.
        layers_before_pool
            Layers before pooling. Either a sequence of tuples with layer type and parameters or
            a dictionary with input-specific layers.
        layers_after_pool
            Layers after pooling.
        cond_output_dropout
            Dropout rate for the last layer of the condition encoder.
        condition_encoder_kwargs
            Keyword arguments for the condition encoder.
        prophet_config
            Configuration for Prophet encoder if using condition_mode='prophet'.
        act_fn
            Activation function.
        time_freqs
            Frequency of the cyclical time encoding.
        time_encoder_dims
            Dimensions of the time embedding.
        time_encoder_dropout
            Dropout rate for the time embedding.
        hidden_dims
            Dimensions of the hidden layers.
        hidden_dropout
            Dropout rate for the hidden layers.
        decoder_dims
            Dimensions of the output layers.
        decoder_dropout
            Dropout rate for the output layers.
        layer_norm_before_concatenation
            If :obj:`True`, applies layer normalization before concatenating
            the embedded time, embedded data, and condition embeddings.
        linear_projection_before_concatenation
            If :obj:`True`, applies a linear projection before concatenating
            the embedded time, embedded data.

    Returns
    -------
        Output of the neural vector field.
    """

    output_dim: int
    max_combination_length: int
    condition_mode: Literal["deterministic", "stochastic", "prophet"] = "deterministic"
    regularization: float = 1.0
    encode_conditions: bool = True
    condition_embedding_dim: int = 32
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=lambda: [])
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    layers_before_pool: Layers_separate_input_t | Layers_t = dc_field(default_factory=lambda: [])
    layers_after_pool: Layers_t = dc_field(default_factory=lambda: [])
    cond_output_dropout: float = 0.0
    mask_value: float = 0.0
    condition_encoder_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    prophet_config: Optional[dict[str, Any]] = None
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    time_freqs: int = 1024
    time_encoder_dims: Sequence[int] = (1024, 1024, 1024)
    time_encoder_dropout: float = 0.0
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    hidden_dropout: float = 0.0
    decoder_dims: Sequence[int] = (1024, 1024, 1024)
    decoder_dropout: float = 0.0
    layer_norm_before_concatenation: bool = False
    linear_projection_before_concatenation: bool = False

    def setup(self):
        """Initialize the network."""
        if self.encode_conditions:
            if self.condition_mode == "prophet":
                if not _PROPHET_AVAILABLE:
                    raise ImportError("Prophet adapter is not available. Please install the prophet_adapter module.")
                if self.prophet_config is None:
                    raise ValueError("prophet_config must be provided when condition_mode='prophet'")

                # Initialize Prophet encoder
                self.condition_encoder = ProphetEncoder(
                    output_dim=self.condition_embedding_dim,
                    tokenizer_config=self.prophet_config["tokenizer_config"],
                    tokenizer_mapping=self.prophet_config["tokenizer_mapping"],
                    learnable_columns=self.prophet_config["learnable_columns"],
                    model_dim=self.prophet_config["model_dim"],
                    num_heads=self.prophet_config["num_heads"],
                    num_layers=self.prophet_config["num_layers"],
                    dropout=self.prophet_config.get("dropout", 0.0),
                )
            else:
                # Original CellFlow condition encoder
                self.condition_encoder = ConditionEncoder(
                    condition_mode=self.condition_mode,
                    regularization=self.regularization,
                    output_dim=self.condition_embedding_dim,
                    pooling=self.pooling,
                    pooling_kwargs=self.pooling_kwargs,
                    layers_before_pool=self.layers_before_pool,
                    layers_after_pool=self.layers_after_pool,
                    covariates_not_pooled=self.covariates_not_pooled,
                    mask_value=self.mask_value,
                    **self.condition_encoder_kwargs,
                )

        self.layer_cond_output_dropout = nn.Dropout(rate=self.cond_output_dropout)
        self.layer_norm_condition = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.time_encoder = MLPBlock(
            dims=self.time_encoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.time_encoder_dropout,
            act_last_layer=False,
        )
        self.layer_norm_time = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.x_encoder = MLPBlock(
            dims=self.hidden_dims,
            act_fn=self.act_fn,
            dropout_rate=self.hidden_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )
        self.layer_norm_x = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )

        self.output_layer = nn.Dense(self.output_dim)

    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        squeeze = x_t.ndim == 1
        if not self.encode_conditions:
            cond_embedding = jnp.concatenate(list(cond.values()), axis=-1)
            cond_mean = cond_embedding
            cond_logvar = jnp.zeros_like(cond_embedding)
        else:
            if self.condition_mode == "prophet":
                # Prophet encoder expects specific input format
                cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
                cond_embedding = cond_mean  # Prophet doesn't use the stochastic approach
            else:
                # Original CellFlow condition encoder
                cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
                if self.condition_mode == "deterministic":
                    cond_embedding = cond_mean
                else:
                    cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)

        cond_embedding = self.layer_cond_output_dropout(cond_embedding, deterministic=not train)

        t_encoded = time_encoder.cyclical_time_encoder(t, n_freqs=self.time_freqs)
        t_encoded = self.time_encoder(t_encoded, training=train)
        x_encoded = self.x_encoder(x_t, training=train)

        t_encoded = self.layer_norm_time(t_encoded)
        x_encoded = self.layer_norm_x(x_encoded)
        cond_embedding = self.layer_norm_condition(cond_embedding)

        if squeeze:
            cond_embedding = jnp.squeeze(cond_embedding)  # , 0)
        elif cond_embedding.shape[0] != x_t.shape[0]:  # type: ignore[attr-defined]
            cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))

        if t_encoded.ndim == 1:
            t_encoded = jnp.reshape(t_encoded, (-1, t_encoded.shape[0]))
        concatenated = jnp.concatenate((t_encoded, x_encoded, cond_embedding), axis=-1)
        out = self.decoder(concatenated, training=train)
        return self.output_layer(out), cond_mean, cond_logvar

    def get_condition_embedding(self, condition: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get the embedding of the condition.

        Parameters
        ----------
            condition
                Conditioning vector of shape ``[batch, ...]``.

        Returns
        -------
            Learnt mean and log-variance of the condition embedding.
            If :attr:`cellflow.model.CellFlow.condition_mode` is ``'deterministic'`` or ``'prophet'``,
            the log-variance is set to zero.
        """
        if self.encode_conditions:
            if self.condition_mode == "prophet":
                # Prophet encoder returns mean and a dummy logvar
                condition_mean, _ = self.condition_encoder(condition, training=False)
                condition_logvar = jnp.zeros_like(condition_mean)
            else:
                # Original CellFlow condition encoder
                condition_mean, condition_logvar = self.condition_encoder(condition, training=False)
        else:
            condition_concat = jnp.concatenate(list(condition.values()), axis=-1)
            logger.warning("Condition encoder is not defined. Returning concatenated input as the embedding.")
            condition_mean = condition_concat
            condition_logvar = jnp.zeros_like(condition_mean)

        return condition_mean, condition_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        """Create initial training state."""
        rng_cond, rng_dropout = jax.random.split(rng)
        # Initialize with a single example
        t_example = jnp.zeros((1,))
        x_example = jnp.zeros((1, input_dim))
        encoder_noise = jnp.zeros((1, self.condition_embedding_dim))

        # Make a single example for each condition
        example_conditions = {}
        for k, v in conditions.items():
            example_conditions[k] = jnp.zeros((1,) + v.shape[1:])

        params = self.init(
            {"params": rng, "dropout": rng_dropout, "condition_encoder": rng_cond},
            t_example,
            x_example,
            example_conditions,
            encoder_noise,
            train=False,
        )["params"]

        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)

    @property
    def output_dims(self):
        """Dimensions of the output layers."""
        return tuple(self.decoder_dims) + (self.output_dim,)

    @property
    def time_encoder(self):
        """The time encoder used."""
        return self._time_encoder

    @time_encoder.setter
    def time_encoder(self, encoder):
        """Set the time encoder."""
        self._time_encoder = encoder

    @property
    def x_encoder(self):
        """The x encoder used."""
        return self._x_encoder

    @x_encoder.setter
    def x_encoder(self, encoder):
        """Set the x encoder."""
        self._x_encoder = encoder

    @property
    def decoder(self):
        """The decoder used."""
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        """Set the decoder."""
        self._decoder = decoder


class GENOTConditionalVelocityField(ConditionalVelocityField):
    """Parameterized neural vector field with conditions for GENOT.

    Parameters
    ----------
        output_dim
            Dimensionality of the output.
        max_combination_length
            Maximum number of covariates in a combination.
        condition_mode
            Mode of the encoder, should be one of:
            - ``'deterministic'``: Learns condition encoding point-wise.
            - ``'stochastic'``: Learns a Gaussian distribution for representing conditions.
        regularization
            Regularization strength in the latent space:
            - For deterministic mode, it is the strength of the L2 regularization.
            - For stochastic mode, it is the strength of the KL divergence regularization.
        encode_conditions
                Processes the embedding of the perturbation conditions if :obj:`True`. If
                :obj:`False`, directly inputs the embedding of the perturbation conditions to the
                generative velocity field. In the latter case, ``'condition_embedding_dim'``,
                ``'condition_encoder_kwargs'``, ``'pooling'``, ``'pooling_kwargs'``,
                ``'layers_before_pool'``, ``'layers_after_pool'``, ``'cond_output_dropout'``
                are ignored.
        condition_embedding_dim
            Dimensions of the condition embedding.
        covariates_not_pooled
            Covariates that will escape pooling (should be identical across all set elements).
        pooling
            Pooling method.
        pooling_kwargs
            Keyword arguments for the pooling method.
        layers_before_pool
            Layers before pooling. Either a sequence of tuples with layer type and parameters or
            a dictionary with input-specific layers.
        layers_after_pool
            Layers after pooling.
        cond_output_dropout
            Dropout rate for the last layer of the condition encoder.
        condition_encoder_kwargs
            Keyword arguments for the condition encoder.
        act_fn
            Activation function.
        time_freqs
            Frequency of the cyclical time encoding.
        time_encoder_dims
            Dimensions of the time embedding.
        time_encoder_dropout
            Dropout rate for the time embedding.
        hidden_dims
            Dimensions of the hidden layers.
        hidden_dropout
            Dropout rate for the hidden layers.
        decoder_dims
            Dimensions of the output layers.
        decoder_dropout
            Dropout rate for the output layers.
        genot_source_dims
            Dimensions of the layers processing the source cells.
        genot_source_dropout
            Dropout rate for the layers processing the source cells.
        layer_norm_before_concatenation
            If :obj:`True`, applies layer normalization before concatenating
            the embedded time, embedded data, and condition embeddings.
        linear_projection_before_concatenation
            If :obj:`True`, applies a linear projection before concatenating
            the embedded time, embedded data.

    Returns
    -------
        Output of the neural vector field.
    """

    output_dim: int
    max_combination_length: int
    condition_mode: Literal["deterministic", "stochastic"] = "deterministic"
    regularization: float = 1.0
    encode_conditions: bool = True
    condition_embedding_dim: int = 32
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=lambda: [])
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    layers_before_pool: Layers_separate_input_t | Layers_t = dc_field(default_factory=lambda: [])
    layers_after_pool: Layers_t = dc_field(default_factory=lambda: [])
    cond_output_dropout: float = 0.0
    mask_value: float = 0.0
    condition_encoder_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    time_freqs: int = 1024
    time_encoder_dims: Sequence[int] = (1024, 1024, 1024)
    time_encoder_dropout: float = 0.0
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    hidden_dropout: float = 0.0
    decoder_dims: Sequence[int] = (1024, 1024, 1024)
    decoder_dropout: float = 0.0
    genot_source_dims: Sequence[int] = (1024, 1024, 1024)
    genot_source_dropout: float = 0.0
    layer_norm_before_concatenation: bool = False
    linear_projection_before_concatenation: bool = False

    def setup(self):
        """Initialize the network."""
        if self.encode_conditions:
            self.condition_encoder = ConditionEncoder(
                condition_mode=self.condition_mode,
                regularization=self.regularization,
                output_dim=self.condition_embedding_dim,
                pooling=self.pooling,
                pooling_kwargs=self.pooling_kwargs,
                layers_before_pool=self.layers_before_pool,
                layers_after_pool=self.layers_after_pool,
                output_dropout=self.cond_output_dropout,
                covariates_not_pooled=self.covariates_not_pooled,
                mask_value=self.mask_value,
                **self.condition_encoder_kwargs,
            )
        self.layer_cond_output_dropout = nn.Dropout(rate=self.cond_output_dropout)
        self.layer_norm_condition = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.time_encoder = MLPBlock(
            dims=self.time_encoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.time_encoder_dropout,
            act_last_layer=False,
        )
        self.layer_norm_time = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.x_encoder = MLPBlock(
            dims=self.hidden_dims,
            act_fn=self.act_fn,
            dropout_rate=self.hidden_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )
        self.layer_norm_x = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.x_0_encoder = MLPBlock(
            dims=self.genot_source_dims,
            act_fn=self.act_fn,
            dropout_rate=self.genot_source_dropout,
        )
        self.layer_norm_x_0 = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )

        self.output_layer = nn.Dense(self.output_dim)

    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        x_0: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
    ):
        squeeze = x_t.ndim == 1
        if not self.encode_conditions:
            cond_embedding = jnp.concatenate(list(cond.values()), axis=-1)
        else:
            cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
            if self.condition_mode == "deterministic":
                cond_embedding = cond_mean
            else:
                cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)
        cond_embedding = self.layer_cond_output_dropout(cond_embedding, deterministic=not train)
        t_encoded = time_encoder.cyclical_time_encoder(t, n_freqs=self.time_freqs)
        t_encoded = self.time_encoder(t_encoded, training=train)
        x_encoded = self.x_encoder(x_t, training=train)
        x_0_encoded = self.x_0_encoder(x_0, training=train)

        t_encoded = self.layer_norm_time(t_encoded)
        x_encoded = self.layer_norm_x(x_encoded)
        x_0_encoded = self.layer_norm_x_0(x_0_encoded)
        cond_embedding = self.layer_norm_condition(cond_embedding)

        if squeeze:
            cond_embedding = jnp.squeeze(cond_embedding)  # , 0)
        elif cond_embedding.shape[0] != x_t.shape[0]:  # type: ignore[attr-defined]
            cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))

        concatenated = jnp.concatenate((t_encoded, x_encoded, x_0_encoded, cond_embedding), axis=-1)
        out = self.decoder(concatenated, training=train)
        return self.output_layer(out), cond_mean, cond_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        """Create the training state.

        Parameters
        ----------
            rng
                Random number generator.
            optimizer
                Optimizer.
            input_dim
                Dimensionality of the velocity field.
            conditions
                Conditions describing the perturbation.

        Returns
        -------
            The training state.
        """
        t, x_t, x_0 = jnp.ones((1, 1)), jnp.ones((1, input_dim)), jnp.ones((1, input_dim))
        encoder_noise = jnp.ones((1, self.condition_embedding_dim))
        cond = {
            pert_cov: jnp.ones((1, self.max_combination_length, condition.shape[-1]))
            for pert_cov, condition in conditions.items()
        }
        params_rng, condition_encoder_rng = jax.random.split(rng, 2)
        params = self.init(
            {"params": params_rng, "condition_encoder": condition_encoder_rng},
            t=t,
            x_t=x_t,
            x_0=x_0,
            cond=cond,
            encoder_noise=encoder_noise,
            train=False,
        )["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
