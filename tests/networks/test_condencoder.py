import jax
import jax.numpy as jnp
import optax
import pytest

import cfp

cond = {
    "pert1": jnp.ones((1, 3, 3)),
    "pert2": jnp.ones((1, 3, 10)),
    "pert3": jnp.ones((1, 3, 5)),
}
cond = {k: v.at[0, 2, :].set(0.0) for k, v in cond.items()}

layers_before_pool = [
    {
        "pert1": (
            ("mlp", {"dims": (32, 32)}),
            ("self_attention", {"num_heads": 4, "qkv_dim": 32}),
        ),
        "pert2": (("mlp", {"dims": (32, 32)}),),
        "pert3": (),
    },
    (
        ("mlp", {"dims": (32, 32)}),
        ("self_attention", {"num_heads": 4, "qkv_dim": 32, "transformer_block": True}),
    ),
    (),
]
layers_after_pool = [
    (("mlp", {"dims": (32, 32)}),),
    (("mlp", {"dims": (32, 32)}), ("self_attention", {"num_heads": 4, "qkv_dim": 12})),
    (),
]


class TestConditionEncoder:
    @pytest.mark.parametrize("pooling", ["mean", "attention_token", "attention_seed"])
    @pytest.mark.parametrize("layers_before_pool", layers_before_pool)
    @pytest.mark.parametrize("layers_after_pool", layers_after_pool)
    def test_velocity_field_init(self, pooling, layers_before_pool, layers_after_pool):
        cond_encoder = cfp.networks.ConditionEncoder(
            output_dim=5,
            pooling=pooling,
            layers_before_pool=layers_before_pool,
            layers_after_pool=layers_after_pool,
            output_dropout=0.1,
        )

        encoder_rng = jax.random.PRNGKey(111)
        opt = optax.adam(1e-3)
        encoder_state = cond_encoder.create_train_state(encoder_rng, opt, cond)
        cond_out = encoder_state.apply_fn(
            {"params": encoder_state.params}, cond, training=True
        )
        assert cond_out.shape == (1, 5)
