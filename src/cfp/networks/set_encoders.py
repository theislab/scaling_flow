from collections.abc import Callable
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.linen import initializers
from flax.training import train_state

__all__ = [
    "MultiHeadAttention",
    "MLPEncoder",
    "DeepSet",
    "DeepSetEncoder",
    "MAB",
    "SAB",
    "PMA",
    "SetTransformer",
    "SetEncoder",
    "ConditionSetEncoder",
]

Shape = tuple[int, ...]


class MultiHeadAttention(nn.Module):
    """Multi-head attention which aggregates sets by learning a token.

    Args:
        num_heads: Number of heads.
        qkv_feature_dim: Feature dimension for the query, key, and value.
        max_comb_length: Maximum number of elements in the set.
        dropout_rate: Dropout rate.
    """

    num_heads: int
    qkv_feature_dim: int
    max_comb_length: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, condition: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Call the module."""
        token_shape = (len(condition), 1) if condition.ndim > 2 else (1,)
        class_token = nn.Embed(num_embeddings=1, features=condition.shape[-1])(
            jnp.int32(jnp.zeros(token_shape))
        )

        condition = jnp.concatenate((class_token, condition), axis=-2)
        mask = self.get_masks(condition)

        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_feature_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )
        emb = attention(condition, mask=mask)
        condition = emb[:, 0, :]  # only continue with token 0

        for cond_dim in self.condition_dims:
            condition = self.act_fn(nn.Dense(cond_dim)(condition))
            condition = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(
                condition
            )
        return condition


class MLPEncoder(nn.Module):
    """A MLP that encodes a condition into a latent vector.

    Args:
        hidden_dim: sequence specifying size of hidden dimensions. If None, the
            encoder a identity function.
        output_dim: output dimension of the latent vector
        act_fn: Activation function
    """

    hidden_dim: int = 128
    n_hidden: int = 1
    output_dim: int = 5
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    dropout_rate: float = 0.0
    training: bool | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool | None = None) -> jnp.ndarray:
        """Call the module."""
        training = nn.merge_param("training", self.training, training)
        if self.hidden_dim is None:
            return x

        squeeze = x.ndim == 1
        if squeeze:
            x = jnp.expand_dims(x, 0)

        z = x
        for _ in range(self.n_hidden):
            Wx = nn.Dense(self.hidden_dim, use_bias=True)
            z = self.act_fn(Wx(z))
            z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=not training)
        Wx = nn.Dense(self.output_dim, use_bias=True)
        z = Wx(z)

        return z.squeeze(0) if squeeze else z

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int | tuple[int, ...],
        **kwargs: Any,
    ):
        """Create initial training state."""
        params = self.init(rng, x=jnp.ones(input_dim), training=False)["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            **kwargs,
        )


class DeepSet(nn.Module):
    """DeepSet layer mapping shape (input_dim, n) to (output_dim, n).

    Args:
        input_dim: input dimension of the latent vector
        output_dim: output dimension of the latent vector
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        alpha_init: initializer function for alpha.
        beta_init: initializer function for beta.
        gamma_init: initializer function for gamma.
    """

    input_dim: int
    output_dim: int
    dtype: Any | None = None
    param_dtype: Any = jnp.float32
    alpha_init: Callable[[Any, Shape, Any], jnp.ndarray] = initializers.lecun_normal()
    beta_init: Callable[[Any, Shape, Any], jnp.ndarray] = initializers.lecun_normal()
    gamma_init: Callable[[Any, Shape, Any], jnp.ndarray] = initializers.lecun_normal()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Call the module."""
        alpha = self.param(
            "alpha",
            self.alpha_init,
            (self.output_dim, self.input_dim),
            self.param_dtype,
        )
        beta = self.param(
            "beta",
            self.beta_init,
            (self.output_dim, self.input_dim),
            self.param_dtype,
        )
        gamma = self.param(
            "gamma",
            self.gamma_init,
            (self.output_dim, 1),
            self.param_dtype,
        )
        y = (
            jnp.einsum("...jz, ij -> ...iz", x, alpha)
            + jnp.einsum("...jz, ij -> ...iz", x.sum(axis=-1)[..., None], beta)
            + gamma
        )
        return y


class DeepSetEncoder(nn.Module):
    """A DeepSet encoder that encodes a condition into a latent vector.

    Args:
        hidden_dim: sequence specifying size of hidden dimensions.
        output_dim: output dimension of the latent vector
        act_fn: Activation function
    """

    hidden_dim_before_pool: int = 128
    hidden_dim_after_pool: int = 128
    n_layers_before_pool: int = 1
    n_layers_after_pool: int = 1
    output_dim: int = 5
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    dropout_rate_before_pool: float = 0.0
    dropout_rate_after_pool: float = 0.0
    equivar_transform: Literal["deepset", "mlp"] = "mlp"
    pool: Literal["max", "mean", "sum"] = "mean"
    training: bool | None = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cond_sizes: jnp.ndarray | None = None,
        training: bool | None = None,
    ) -> jnp.ndarray:
        """Call the module."""
        # prepare input and mask
        training = nn.merge_param("training", self.training, training)
        squeeze = x.ndim == 2
        if squeeze:
            x = jnp.expand_dims(x, 0)
        if cond_sizes is None:
            cond_sizes = jnp.full(x.shape[0], x.shape[1])
        mask = jnp.arange(x.shape[1]) < cond_sizes[:, None]
        z = x

        # equivariant transform
        if self.equivar_transform == "deepset":
            z = z.transpose(0, 2, 1)
            for _ in range(self.n_layers_before_pool):
                Wx = DeepSet(z.shape[1], self.hidden_dim_before_pool)
                z = self.act_fn(Wx(z))
                z = nn.Dropout(rate=self.dropout_rate_before_pool)(
                    z, deterministic=not training
                )
            axis_pool = 2
            mask = jnp.expand_dims(mask, 1)
        elif self.equivar_transform == "mlp":
            for _ in range(self.n_layers_before_pool):
                Wx = nn.Dense(self.hidden_dim_before_pool, use_bias=True)
                z = Wx(z)
                z = self.act_fn(z)
                z = nn.Dropout(rate=self.dropout_rate_before_pool)(
                    z, deterministic=not training
                )
            axis_pool = 1
            mask = jnp.expand_dims(mask, -1)
        else:
            raise ValueError("Invalid equivariant transform")

        # pooling
        if self.pool == "max":
            z = z + (1 - mask) * (-99999)
            z = z.max(axis=axis_pool)
        elif self.pool == "mean":
            z = z * mask
            z = z.mean(axis=axis_pool)
        elif self.pool == "sum":
            z = z * mask
            z = z.sum(axis=axis_pool)

        # mlp
        for _ in range(self.n_layers_after_pool):
            Wx = nn.Dense(self.hidden_dim_after_pool, use_bias=True)
            z = Wx(z)
            z = self.act_fn(z)
            z = nn.Dropout(rate=self.dropout_rate_after_pool)(
                z, deterministic=not training
            )
        Wx = nn.Dense(self.output_dim, use_bias=True)
        z = Wx(z)

        return z.squeeze(0) if squeeze else z


class MAB(nn.Module):
    """Multi-head attention block."""

    dim_V: int
    num_heads: int
    ln: bool = False
    dropout_rate: float = 0.0
    training: bool | None = None

    @nn.compact
    def __call__(self, Q, K, mask=None, training: bool | None = None):
        """Call the module."""
        training = nn.merge_param("training", self.training, training)
        is_eval = not training

        Q = nn.Dense(self.dim_V)(Q)
        K, V = nn.Dense(self.dim_V)(K), nn.Dense(self.dim_V)(K)

        Q_ = jnp.concatenate(jnp.split(Q, self.num_heads, axis=2), axis=0)
        K_ = jnp.concatenate(jnp.split(K, self.num_heads, axis=2), axis=0)
        V_ = jnp.concatenate(jnp.split(V, self.num_heads, axis=2), axis=0)

        A = jnp.matmul(Q_, K_.transpose(0, 2, 1)) / jnp.sqrt(self.dim_V)
        if mask is not None:
            if Q_.shape[1] != 1:
                # mask from (batch_, set_, set_) to (batch_ * num_heads, set_, set_)
                mask = jnp.repeat(mask, self.num_heads, axis=0)
            else:
                # mask from (batch_, set_, set_) to (batch_ * num_heads, 1, set_)
                mask = jnp.repeat(mask[:, [0], :], self.num_heads, axis=0)
            A = jnp.where(mask, A, -1e9)
        A = nn.softmax(A)

        O = jnp.concatenate(
            jnp.split(Q_ + jnp.matmul(A, V_), self.num_heads, axis=0), axis=2
        )

        O = nn.Dropout(rate=self.dropout_rate)(O, deterministic=is_eval)
        if self.ln:
            O = nn.LayerNorm()(O)
        O_ = nn.relu(nn.Dense(self.dim_V)(O))
        O_ = nn.Dropout(rate=self.dropout_rate)(O_, deterministic=is_eval)
        O = O + O_
        if self.ln:
            O = nn.LayerNorm()(O)

        return O


class SAB(nn.Module):
    """Self-attention block."""

    dim_out: int
    num_heads: int
    ln: bool = False
    dropout_rate: float = 0.0
    training: bool | None = None

    @nn.compact
    def __call__(self, X, mask=None, training: bool | None = None):
        """Call the module."""
        training = nn.merge_param("training", self.training, training)
        return MAB(self.dim_out, self.num_heads, self.ln, self.dropout_rate, training)(
            X, X, mask
        )


class PMA(nn.Module):
    """Pooling by multi-head attention."""

    dim: int
    num_heads: int
    num_seeds: int
    ln: bool = False
    dropout_rate: float = 0.0
    training: bool | None = None

    @nn.compact
    def __call__(self, X, mask=None, training: bool | None = None):
        """Call the module."""
        training = nn.merge_param("training", self.training, training)
        S = self.param(
            "S", initializers.xavier_uniform(), (1, self.num_seeds, self.dim)
        )
        S = jnp.tile(S, (X.shape[0], 1, 1))
        return MAB(self.dim, self.num_heads, self.ln, self.dropout_rate, training)(
            S, X, mask
        )


class SetTransformer(nn.Module):
    """
    Set transformer.

    Parameters
    ----------
    dim_input : int
        Input dimension.
    num_outputs : int
        Number of output features.
    dim_output : int
        Output dimension.
    num_inds : int
        Number of induced set elements.
    dim_hidden : int
        Hidden dimension.
    num_heads : int
        Number of heads.
    ln : bool
        Whether to use layer normalization.
    """

    # Copyright (c) 2020 Juho Lee
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    dim_output: int
    num_outputs: int = 1
    dim_hidden_before_pool: int = 128
    dim_hidden_after_pool: int = 128
    num_heads: int = 4
    ln: bool = False
    n_enc_sab: int = 2
    n_dec_sab: int = 2
    dropout_rate_before_pool: float = 0.0
    dropout_rate_pool: float = 0.0
    dropout_rate_after_pool: float = 0.0
    training: bool | None = None

    def setup(self):
        """Setup the module."""
        self.enc_layers = [
            SAB(
                self.dim_hidden_before_pool,
                self.num_heads,
                self.ln,
                self.dropout_rate_before_pool,
            )
            for _ in range(self.n_enc_sab)
        ]
        self.dec_layers = (
            [
                PMA(
                    self.dim_hidden_after_pool,
                    self.num_heads,
                    self.num_outputs,
                    self.ln,
                    self.dropout_rate_pool,
                )
            ]
            + [
                SAB(
                    self.dim_hidden_after_pool,
                    self.num_heads,
                    self.ln,
                    self.dropout_rate_after_pool,
                )
                for _ in range(self.n_dec_sab)
            ]
            + [nn.Dense(self.dim_output)]
        )

    def __call__(
        self,
        x: jnp.ndarray,
        cond_sizes: jnp.ndarray | None = None,
        training: bool | None = None,
    ) -> jnp.ndarray:
        """Call the module."""
        training = nn.merge_param("training", self.training, training)
        squeeze = x.ndim == 2
        if squeeze:
            x = jnp.expand_dims(x, 0)

        if cond_sizes is None:
            cond_sizes = jnp.full(x.shape[0], x.shape[1])

        # create mask, transform it from (batch_, set_) to (batch_, set_, set_) for attention
        mask = jnp.arange(x.shape[1]) < cond_sizes[:, None]
        mask = jnp.expand_dims(mask, -1)
        mask = mask & mask.transpose(0, 2, 1)

        for layer in self.enc_layers:
            x = layer(x, mask, training)
        for layer in self.dec_layers:
            if isinstance(layer, PMA):
                x = layer(x, mask, training)
            elif isinstance(layer, SAB):
                x = layer(x, training=training)
            else:
                x = layer(x)

        x = x.squeeze(1)

        return x.squeeze(0) if squeeze else x


class SetEncoder(nn.Module):
    """
    Encoder for conditions represented as sets of perturbations, times, and doses.

    Parameters
    ----------
    hidden_dim : Sequence[int]
        Sequence specifying size of hidden dimensions.
    max_set_size : int
        Maximum set size, to which all sets were padded.
    output_dim : int
        Output dimension of the latent vector.
    act_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Activation function.
    set_encoder : Literal["deepset", "transformer"]
        Set encoder to use.
    set_encoder_kwargs : dict
        Keyword arguments for the set encoder.
    perturb_encoder_hidden_dim : Optional[Sequence[int]]
        Hidden dimensions for the perturbation encoder (if any).
    perturb_encoder_output_dim : Optional[int]
        Output dimension for the perturbation encoder (if any).
    time_encoder : Optional[Callable[[jnp.ndarray], jnp.ndarray]]
        Encoder for time.
    dose_encoder : Optional[Callable[[jnp.ndarray], jnp.ndarray]]
        Encoder for dose.
    """

    hidden_dim_before_pool: int = 128
    hidden_dim_after_pool: int = 128
    n_layers_before_pool: int = 1
    n_layers_after_pool: int = 1
    max_set_size: int = 2
    output_dim: int = 10
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    dropout_rate_before_pool: float = 0.0
    dropout_rate_after_pool: float = 0.0
    set_encoder: Literal["deepset", "transformer"] = "deepset"
    set_encoder_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    equivar_transform: Literal["deepset", "mlp"] = "mlp"
    training: bool | None = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cond_sizes: jnp.ndarray | None = None,
        training: bool | None = None,
    ) -> jnp.ndarray:
        """
        Apply the set encoder.

        Parameters
        ----------
        x : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            tuple of perturbations, times, doses, and condition sizes.
        """
        training = nn.merge_param("training", self.training, training)
        if self.set_encoder == "deepset":
            SetEncoder = DeepSetEncoder(
                hidden_dim_before_pool=self.hidden_dim_before_pool,
                hidden_dim_after_pool=self.hidden_dim_after_pool,
                n_layers_before_pool=self.n_layers_before_pool,
                n_layers_after_pool=self.n_layers_after_pool,
                output_dim=self.output_dim,
                act_fn=self.act_fn,
                dropout_rate_before_pool=self.dropout_rate_before_pool,
                dropout_rate_after_pool=self.dropout_rate_after_pool,
                equivar_transform=self.equivar_transform,
                **self.set_encoder_kwargs,
            )
        elif self.set_encoder == "transformer":
            SetEncoder = SetTransformer(
                dim_output=self.output_dim,
                dim_hidden_before_pool=self.hidden_dim_before_pool,
                dim_hidden_after_pool=self.hidden_dim_after_pool,
                n_enc_sab=self.n_layers_before_pool,
                n_dec_sab=self.n_layers_after_pool,
                dropout_rate_before_pool=self.dropout_rate_before_pool,
                dropout_rate_after_pool=self.dropout_rate_after_pool,
                **self.set_encoder_kwargs,
            )
        else:
            raise ValueError(f"Set encoder {self.set_encoder} not implemented.")
        z = SetEncoder(x, cond_sizes, training)

        return z

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        **kwargs: Any,
    ):
        """Create initial training state."""
        params = self.init(
            rng,
            x=jnp.ones((1, self.max_set_size, input_dim)),
            cond_sizes=jnp.array([1]),
            training=False,
        )["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            **kwargs,
        )


class ConditionSetEncoder(nn.Module):
    """
    Encoder for conditions represented as sets of perturbations, times, and doses.

    Parameters
    ----------
    hidden_dim : Sequence[int]
        Sequence specifying size of hidden dimensions.
    max_set_size : int
        Maximum set size, to which all sets were padded.
    output_dim : int
        Output dimension of the latent vector.
    act_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Activation function.
    set_encoder : Literal["deepset", "transformer"]
        Set encoder to use.
    set_encoder_kwargs : dict
        Keyword arguments for the set encoder.
    perturb_encoder_hidden_dim : Optional[Sequence[int]]
        Hidden dimensions for the perturbation encoder (if any).
    perturb_encoder_output_dim : Optional[int]
        Output dimension for the perturbation encoder (if any).
    time_encoder : Optional[Callable[[jnp.ndarray], jnp.ndarray]]
        Encoder for time.
    dose_encoder : Optional[Callable[[jnp.ndarray], jnp.ndarray]]
        Encoder for dose.
    """

    hidden_dim_before_pool: int = 128
    hidden_dim_after_pool: int = 128
    n_layers_before_pool: int = 1
    n_layers_after_pool: int = 1
    max_set_size: int = 2
    output_dim: int = 10
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    dropout_rate_before_pool: float = 0.0
    dropout_rate_after_pool: float = 0.0
    set_encoder: Literal["deepset", "transformer"] = "deepset"
    set_encoder_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    equivar_transform: Literal["deepset", "mlp"] = "mlp"
    training: bool | None = None

    @nn.compact
    def __call__(
        self,
        x: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        training: bool,
    ) -> jnp.ndarray:
        """
        Apply the set encoder.

        Parameters
        ----------
        x : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            tuple of perturbations, times, doses, and condition sizes.
        """
        training = nn.merge_param("training", self.training, training)
        perturbs, times, doses, cond_sizes = x
        CondSetEncoder = SetEncoder(
            hidden_dim_before_pool=self.hidden_dim_before_pool,
            hidden_dim_after_pool=self.hidden_dim_after_pool,
            n_layers_before_pool=self.n_layers_before_pool,
            n_layers_after_pool=self.n_layers_after_pool,
            output_dim=self.output_dim,
            act_fn=self.act_fn,
            dropout_rate_before_pool=self.dropout_rate_before_pool,
            dropout_rate_after_pool=self.dropout_rate_after_pool,
            set_encoder=self.set_encoder,
            set_encoder_kwargs=self.set_encoder_kwargs,
            equivar_transform=self.equivar_transform,
        )
        # NOTE: maybe better to format the perturbs * concentrations earlier
        perturb_doses = perturbs * doses
        conds = jnp.concatenate([perturb_doses, times], axis=2)
        z = CondSetEncoder(conds, cond_sizes, training)

        return z

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: tuple[int, ...],
        **kwargs: Any,
    ):
        """Create initial training state."""
        params = self.init(
            rng,
            x=(
                jnp.ones((1, self.max_set_size, input_dim[0])),
                jnp.ones((1, self.max_set_size, input_dim[1])),
                jnp.ones((1, self.max_set_size, input_dim[2])),
                jnp.array([1]),
            ),
            training=False,
        )["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            **kwargs,
        )
