import abc
from collections.abc import Callable, Sequence
from dataclasses import field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.linen import initializers
from flax.training import train_state

__all__ = [
    "MLPBlock",
    "SelfAttention",
    "SeedAttentionPooling",
    "TokenAttentionPooling",
    "ConditionEncoder",
]


class BaseModule(abc.ABC, nn.Module):
    """Base module for condition encoder and its components."""

    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Forward pass."""
        pass


class MLPBlock(BaseModule):
    """
    MLP block.

    Attributes
    ----------
    dims : Sequence[int]
        Dimensions of the MLP layers.
    dropout_rate : float
        Dropout rate.
    act_last_layer : bool
        Whether to apply the activation function to the last layer.
    act_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Activation function.
    """

    dims: Sequence[int] = (128, 128, 128)
    dropout_rate: float = 0.0
    act_last_layer: bool = True
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape (batch_size, input_dim).
        training : bool
            Whether the model is in training mode.

        Returns
        -------
        z : jnp.ndarray
            Output tensor of shape (batch_size, output_dim).
        """
        z = x
        for i in range(len(self.dims) - 1):
            z = self.act_fn(nn.Dense(self.dims[i])(z))
            z = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
        z = nn.Dense(self.dims[-1])(z)
        z = self.act_fn(z) if self.act_last_layer else z
        z = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
        return z


class SelfAttention(BaseModule):
    """
    Self-attention optionally followed by FC layer with residual connection, making it a transformer block.

    Attributes
    ----------
    num_heads : int
        Number of heads.
    qkv_dim : int
        Dimensionality of the query, key, and value.
    dropout_rate : float
        Dropout rate.
    transformer_block : bool
        Whether to make it a transformer block (adds FC layer with residual connection).
    layer_norm : bool
        Whether to use layer normalization
    """

    num_heads: int = 8
    qkv_dim: int = 64
    dropout_rate: float = 0.0
    transformer_block: bool = False
    layer_norm: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ):
        """
        Forward pass.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape (batch_size, set_size, input_dim).
        mask : Optional[jnp.ndarray]
            Mask tensor of shape (batch_size, 1 | num_heads, set_size, set_size).
        training : bool
            Whether the model is in training mode.

        Returns
        -------
        z : jnp.ndarray
            Output tensor of shape (batch_size, set_size, input_dim).
        """
        # self-attention
        z = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_dim,
            dropout_rate=self.dropout_rate,
        )(x, mask=mask, deterministic=not training)

        if self.transformer_block:
            # query residual connection
            z = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
            z = z + x
            if self.layer_norm:
                z = nn.LayerNorm()(z)
            # FC layer with residual connection
            z_ = self.act_fn(nn.Dense(self.qkv_dim)(z))
            z_ = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
            z = z + z_

        return z


class SeedAttentionPooling(BaseModule):
    """
    Pooling by multi-head attention with a trainable seed.

    Attributes
    ----------
    num_heads : int
        Number of heads.
    v_dim : int
        Dimensionality of the value.
    seed_dim : int
        Dimensionality of the seed.
    dropout_rate : float
        Dropout rate.
    transformer_block : bool
        Whether to make it a transformer block (adds FC layer with residual connection).
    layer_norm : bool
        Whether to use layer normalization.
    act_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Activation function.

    References
    ----------
    #FIXME: add set transformer reference
    """

    num_heads: int = 8
    v_dim: int = 64
    seed_dim: int = 64
    dropout_rate: float = 0.0
    transformer_block: bool = False
    layer_norm: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ):
        """
        Apply the pooling by multi-head attention.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape (batch_size, set_size, input_dim).
        mask : Optional[jnp.ndarray]
            Mask tensor of shape (batch_size, 1, set_size, set_size).
        training : bool
            Whether the model is in training mode.
        """
        # trainable seed
        S = self.param("S", initializers.xavier_uniform(), (1, 1, self.seed_dim))
        S = jnp.tile(S, (x.shape[0], 1, 1))

        # multi-head attention
        Q = nn.Dense(self.v_dim)(S)
        K, V = nn.Dense(self.v_dim)(x), nn.Dense(self.v_dim)(x)
        Q_ = jnp.concatenate(jnp.split(Q, self.num_heads, axis=2), axis=0)
        K_ = jnp.concatenate(jnp.split(K, self.num_heads, axis=2), axis=0)
        V_ = jnp.concatenate(jnp.split(V, self.num_heads, axis=2), axis=0)
        A = jnp.matmul(Q_, K_.transpose(0, 2, 1)) / jnp.sqrt(self.v_dim)
        if mask is not None:
            # mask from (batch_, 1 | num_heads, set_, set_) to (batch_ * num_heads, 1, set_)
            mask = jnp.repeat(mask[:, 0, [0], :], self.num_heads, axis=0)

            A = jnp.where(mask, A, -1e9)
        A = nn.softmax(A)
        A = jnp.matmul(A, V_)

        if self.transformer_block:
            # query residual connection
            O = jnp.concatenate(jnp.split(Q_ + A, self.num_heads, axis=0), axis=2)
            O = nn.Dropout(rate=self.dropout_rate)(O, deterministic=not training)
            if self.layer_norm:
                O = nn.LayerNorm()(O)
            # FC layer with residual connection
            O_ = self.act_fn(nn.Dense(self.v_dim)(O))
            O_ = nn.Dropout(rate=self.dropout_rate)(O_, deterministic=not training)
            O = O + O_
            if self.layer_norm:
                O = nn.LayerNorm()(O)
        else:
            O = jnp.concatenate(jnp.split(A, self.num_heads, axis=0), axis=2)

        return O.squeeze(1)


class TokenAttentionPooling(BaseModule):
    """
    Multi-head attention which aggregates sets by learning a token.

    Attributes
    ----------
    output_dim : int
        Dimensionality of the output.
    num_heads : int
        Number of heads.
    qkv_feature_dim : int
        Feature dimension for the query, key, and value.
    hidden_dims : Sequence[int]
        Dimensions of the MLP layers after the attention.
    dropout_rate : float
        Dropout rate.
    act_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Activation function.
    """

    num_heads: int = 8
    qkv_dim: int = 64
    dropout_rate: float = 0.0
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape (batch_size, set_size, input_dim).
        mask : Optional[jnp.ndarray]
            Mask tensor of shape (batch_size, 1 | num_heads, set_size, set_size).
        training : bool
            Whether the model is in training mode.
        """
        # add token
        token_shape = (len(x), 1)
        class_token = nn.Embed(num_embeddings=1, features=x.shape[-1])(
            jnp.int32(jnp.zeros(token_shape))
        )
        z = jnp.concatenate((class_token, x), axis=-2)
        token_mask = jnp.zeros((x.shape[0], 1, x.shape[1] + 1, x.shape[1] + 1))
        token_mask = token_mask.at[:, :, 0, :].set(1)
        token_mask = token_mask.at[:, :, :, 0].set(1)
        token_mask = token_mask.at[:, :, 1:, 1:].set(mask)

        # attention
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_dim,
            dropout_rate=self.dropout_rate,
        )
        emb = attention(z, mask=token_mask, deterministic=not training)

        # only continue with token 0
        z = emb[:, 0, :]

        return z


class ConditionEncoder(BaseModule):
    """
    Encoder for conditions represented as sets of perturbations.

    Attributes
    ----------
    output_dim : int
        Dimensionality of the output.
    pooling : Literal["mean", "attention_token", "attention_seed"]
        Pooling method.
    pooling_kwargs : dict
        Keyword arguments for the pooling method.
    layers_before_pool : Sequence[tuple[Literal["mlp", "self_attention"], dict]] | dict
        Layers before pooling. Either a sequence of tuples with layer type and parameters or a dictionary with input-specific layers.
    layers_after_pool : Sequence[tuple[Literal["mlp", "self-attention"], dict]]
        Layers after pooling.
    output_dropout : float
        Dropout rate for the output layer.
    mask_value : float
        Value for masked elements used in input conditions.
    """

    output_dim: int
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict = field(default_factory=lambda: {})
    layers_before_pool: (
        Sequence[tuple[Literal["mlp", "self_attention"], dict]] | dict
    ) = field(default_factory=lambda: [])
    layers_after_pool: Sequence[tuple[Literal["mlp", "self-attention"], dict]] = field(
        default_factory=lambda: []
    )
    output_dropout: float = 0.0
    mask_value: float = 0.0

    def setup(self):
        """Initialize the modules."""
        # modules before pooling
        self.separate_inputs = isinstance(self.layers_before_pool, dict)
        if self.separate_inputs:
            # different layers for different inputs
            self.before_pool_modules = {}
            for cond_key, layers in self.layers_before_pool.items():
                self.before_pool_modules[cond_key] = self._get_layers(layers)
        else:
            self.before_pool_modules = self._get_layers(self.layers_before_pool)

        # pooling
        if self.pooling == "mean":
            self.pool_module = lambda x, mask: jnp.mean(x * mask, axis=-2)
        elif self.pooling == "attention_token":
            self.pool_module = TokenAttentionPooling(**self.pooling_kwargs)
        elif self.pooling == "attention_seed":
            self.pool_module = SeedAttentionPooling(**self.pooling_kwargs)

        # modules after pooling
        self.after_pool_modules = self._get_layers(
            self.layers_after_pool, self.output_dim, self.output_dropout
        )

    def __call__(
        self,
        conditions: dict[str, jnp.ndarray],
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Apply the set encoder.

        Parameters
        ----------
        conditions : dict[str, jnp.ndarray]
            Dictionary of batch of conditions of shape ``(batch_size, set_size, condition_dim)``.
        training : bool
            Whether the model is in training mode.

        Returns
        -------
        Encoded conditions of shape ``(batch_size, output_dim)``.
        """
        mask, attention_mask = self._get_masks(conditions)

        # apply modules before pooling
        if self.separate_inputs:
            processed_inputs = []
            for cond_key, conditions_i in conditions.items():
                conditions_i = self._apply_modules(
                    self.before_pool_modules[cond_key],
                    conditions_i,
                    attention_mask,
                    training,
                )
                processed_inputs.append(conditions_i)
            conditions = jnp.concatenate(processed_inputs, axis=-1)
        else:
            conditions = jnp.concatenate(list(conditions.values()), axis=-1)
            conditions = self._apply_modules(
                self.before_pool_modules, conditions, attention_mask, training
            )

        # pooling
        pool_mask = mask if self.pooling == "mean" else attention_mask
        conditions = self.pool_module(conditions, pool_mask)

        # apply modules after pooling
        conditions = self._apply_modules(
            self.after_pool_modules, conditions, attention_mask, training
        )

        return conditions

    def _get_layers(
        self,
        layers: dict,
        output_dim: int | None = None,
        dropout_rate: float | None = None,
    ) -> list:
        modules = []
        for layer_type, layer_params in layers:
            if layer_type == "mlp":
                layer = MLPBlock(**layer_params)
            elif layer_type == "self_attention":
                layer = SelfAttention(**layer_params)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            modules.append(layer)
        if output_dim is not None:
            modules.append(nn.Dense(output_dim))
            if dropout_rate is not None:
                modules.append(nn.Dropout(dropout_rate))
        return modules

    def _get_masks(
        self,
        conditions: jnp.ndarray | tuple[jnp.ndarray, ...],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get mask for padded conditions tensor."""
        if isinstance(conditions, tuple):
            masks = [jnp.all(c == self.mask_value, axis=-1) for c in conditions]
            if not all(jnp.array_equal(masks[0], mask) for mask in masks):
                raise ValueError("Conditions have different masked values.")

        # mask of shape (batch_size, set_size)
        mask = 1 - jnp.all(conditions == self.mask_value, axis=-1)
        mask = jnp.expand_dims(mask, -1)

        # attention mask of shape (batch_size, 1, set_size, set_size)
        attention_mask = mask & jnp.matrix_transpose(mask)
        attention_mask = jnp.expand_dims(attention_mask, 1)

        return mask, attention_mask

    def _apply_modules(self, modules, conditions, attention_mask, training):
        for module in modules:
            if isinstance(module, SelfAttention):
                conditions = module(conditions, attention_mask, training)
            if isinstance(module, nn.Dense):
                conditions = module(conditions)
            else:
                conditions = module(conditions, training)
        return conditions

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int | tuple[int, ...],
        **kwargs: Any,
    ):
        """Create initial training state."""
        params = self.init(rng, conditions=jnp.empty(input_dim), training=False)[
            "params"
        ]
        return train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            **kwargs,
        )
