import jax
import jax.numpy as jnp
import optax
import pytest

import cfp

x_test = jnp.ones((10, 5)) * 10
t_test = jnp.ones((10, 1))
cond = jnp.ones((10, 2, 3))


class TestVelocityField:
    @pytest.mark.parametrize("decoder_dims", [(32, 32), ()])
    @pytest.mark.parametrize("hidden_dims", [(32, 32), ()])
    @pytest.mark.parametrize("condition_encoder", ["transformer", "deepset"])
    def test_velocity_field_init(self, condition_encoder, hidden_dims, decoder_dims):
        vf = cfp.networks.ConditionalVelocityField(
            condition_encoder=condition_encoder,
            output_dim=5,
            condition_dim=3,
            condition_embedding_dim=12,
            hidden_dims=hidden_dims,
            decoder_dims=decoder_dims,
        )
        assert vf.output_dims == decoder_dims + (5,)

        vf_rng = jax.random.PRNGKey(111)
        opt = optax.adam(1e-3)
        vf_state = vf.create_train_state(vf_rng, opt, 5)
        x_out = vf_state.apply_fn({"params": vf_state.params}, t_test, x_test, cond, train=True)
        assert x_out.shape == (10, 5)

        cond_embed = vf.apply(
            {"params": vf_state.params},
            cond,
            method="encode_conditions",
        )
        assert cond_embed.shape == (10, 12)
