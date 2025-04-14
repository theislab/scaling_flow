from collections.abc import Callable, Sequence
from typing import Any, Dict, Literal, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

from cellflow.networks._utils import MLPBlock

__all__ = ["ProphetEncoder"]

class MultiHeadAttention(nn.Module):
    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0

    def setup(self):
        self.wq = nn.Dense(self.qkv_features)
        self.wk = nn.Dense(self.qkv_features)
        self.wv = nn.Dense(self.qkv_features)
        self.dense = nn.Dense(self.qkv_features)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.shape
        x = x.reshape((batch_size, seq_length, self.num_heads, -1))
        return x.transpose((0, 2, 1, 3))

    def __call__(self, x, attn_mask=None):
        q = self.split_heads(self.wq(x))
        k = self.split_heads(self.wk(x))
        v = self.split_heads(self.wv(x))

        # Scaled dot-product attention
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(self.qkv_features)

        if attn_mask is not None:
            # Apply the attention mask (assuming attn_mask is of shape [batch_size, seq_length])
            attn_weights = jnp.where(attn_mask[:, None, None, :], attn_weights, -jnp.inf)

        attn_weights = nn.softmax(attn_weights)
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Concatenate heads and pass through final dense layer
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape(x.shape)
        return self.dense(attn_output)

class FeedForward(nn.Module):
    hidden_dim: int
    dropout_rate: float = 0.0

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_dim)
        self.dense2 = nn.Dense(int(self.hidden_dim/2)) # assuming ratio 2
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, training):
        x = nn.gelu(self.dense1(x))
        x = self.dropout(x, deterministic=not training)
        return self.dense2(x)

class TransformerEncoderLayer(nn.Module):
    num_heads: int
    qkv_features: int
    hidden_dim: int
    dropout_rate: float = 0.0

    def setup(self):
        self.attention = MultiHeadAttention(self.num_heads, self.qkv_features, self.dropout_rate)
        self.ffn = FeedForward(self.hidden_dim, self.dropout_rate)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

    def __call__(self, x, attn_mask=None, training=False):
        attn_output = self.attention(x, attn_mask)
        x = self.ln1(x + attn_output)  # Residual connection
        ffn_output = self.ffn(x, training)
        return self.ln2(x + ffn_output)  # Residual connection

class TransformerEncoder(nn.Module):
    num_layers: int
    num_heads: int
    qkv_features: int
    hidden_dim: int
    dropout_rate: float = 0.0

    def setup(self):
        self.layers = [TransformerEncoderLayer(self.num_heads, self.qkv_features, self.hidden_dim, self.dropout_rate) for _ in range(self.num_layers)]

    def __call__(self, x, attn_mask=None, training=False):
        for layer in self.layers:
            x = layer(x, attn_mask, training)
        return x

class ProphetEncoder(nn.Module):
    """
    JAX/Flax implementation of Prophet's encoder architecture.

    Parameters
    ----------
    output_dim : int
        Output dimension of the encoder.
    tokenizer_config : Dict[str, Dict]
        Configuration for tokenizer networks.
    tokenizer_mapping : Dict[str, str]
        Mapping from input position to tokenizer name.
    learnable_columns : Sequence[str]
        Names of columns that should use learnable embeddings.
    model_dim : int
        Model dimension for transformer.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer layers.
    dropout : float
        Dropout rate.
    """

    output_dim: int
    tokenizer_config: Dict[str, Dict]
    tokenizer_mapping: Dict[int, str]
    learnable_columns: Sequence[str]
    model_dim: int
    num_heads: int
    num_layers: int
    dropout: float = 0.0
    pool: Literal["cls"] = "cls"

    def setup(self):
        tokenizer_nets = {}
        for name, config in self.tokenizer_config.items():
            input_dim = config.get("input_dim")
            num_layers = config.get("num_layers")
            if not isinstance(input_dim, int) or not isinstance(num_layers, int):
                raise ValueError(f"Invalid configuration for {name}: input_dim and num_layers must be integers.")

            tokenizer_nets[name] = MLPBlock(
                dims=[self.model_dim] * (num_layers - 1) + [self.model_dim],
                dropout_rate=config.get("dropout", 0.0),
                act_fn=nn.gelu,
                act_last_layer=True,
            )

        self.tokenizer_nets = tokenizer_nets

        # Create learnable embeddings
        learnable_embeddings = {}
        for col in self.learnable_columns:
            learnable_embeddings[col] = nn.Embed(
                num_embeddings=1000,  # Could be configurable
                features=self.model_dim,
            )

        self.learnable_embeddings = learnable_embeddings
        # CLS token
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, self.model_dim)
        )

        # Transformer encoder layers
        self.transformer = TransformerEncoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            qkv_features=self.model_dim,
            hidden_dim=2 * self.model_dim,
            dropout_rate=self.dropout,
        )

        # Output network
        # self.output_net = MLPBlock(
        #     dims=[self.model_dim, self.model_dim, self.output_dim],
        #     dropout_rate=self.dropout,
        #     act_fn=nn.gelu,
        #     act_last_layer=False,
        # )

    def __call__(
        self,
        token_dict: Dict[Any, jnp.ndarray],
        attn_mask: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """Forward pass through Prophet encoder.

        Parameters
        ----------
        token_dict : Dict[Any, jnp.ndarray]
            Dictionary of input tokens.
        attn_mask : Optional[jnp.ndarray]
            Attention mask for transformer.
        training : bool
            Whether in training mode.

        Returns
        -------
        jnp.ndarray
            Encoded representation.
        """
        # Process tokens
        processed_tokens = []
        for order, name in self.tokenizer_mapping.items():
            if order in token_dict:
                x = token_dict[order].astype(jnp.float32)
                x_token = self.tokenizer_nets[name](x, training=training)
                #x_token = jnp.expand_dims(x_token, 1)  # Add sequence dimension
                processed_tokens.append(x_token)

        # Process learnable embeddings
        learnable_embs = []
        for col in self.learnable_columns:
            if col in token_dict:
                emb = self.learnable_embeddings[col](token_dict[col])
                #emb = jnp.expand_dims(emb, 1)  # Add sequence dimension
                learnable_embs.append(emb)

        # Prepend CLS token
        batch_size = processed_tokens[0].shape[0] if processed_tokens else token_dict[list(token_dict.keys())[0]].shape[0]
        cls = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.model_dim))

        # Concatenate all tokens
        tokens = jnp.concatenate([cls] + learnable_embs + processed_tokens, axis=1)

        # Create padding mask for transformer if not provided
        if attn_mask is None:
            attn_mask = jnp.zeros((batch_size, tokens.shape[1])).astype(jnp.bool_)

        # Apply transformer (encoder only)
        encoder_outputs = self.transformer(
            tokens,
            attn_mask=attn_mask,
            training=training,
        )

        # Use CLS token output (first token)
        x = encoder_outputs[:, 0, :]

        # Apply output network
        # x = self.output_net(x, training=training)

        return x, jnp.zeros_like(x)  # Return dummy logvar for compatibility