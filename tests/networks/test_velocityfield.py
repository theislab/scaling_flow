import jax
import jax.numpy as jnp
import optax
import pytest

import cfp

x_test = jnp.ones((10, 5)) * 10
t_test = jnp.ones((10, 1))
cond = {"pert1": jnp.ones((1, 2, 3))}


class TestVelocityField:
    @pytest.mark.parametrize("decoder_dims", [(32, 32), ()])
    @pytest.mark.parametrize("hidden_dims", [(32, 32), ()])
    @pytest.mark.parametrize("layer_norm_before_concatenation", [True, False])
    @pytest.mark.parametrize("linear_projection_before_concatenation", [True, False])
    def test_velocity_field_init(
        self,
        hidden_dims,
        decoder_dims,
        layer_norm_before_concatenation,
        linear_projection_before_concatenation,
    ):
        vf = cfp.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=hidden_dims,
            decoder_dims=decoder_dims,
            layer_norm_before_concatenation=layer_norm_before_concatenation,
            linear_projection_before_concatenation=linear_projection_before_concatenation,
        )
        assert vf.output_dims == decoder_dims + (5,)

        vf_rng = jax.random.PRNGKey(111)
        opt = optax.adam(1e-3)
        vf_state = vf.create_train_state(vf_rng, opt, 5, cond)
        x_out = vf_state.apply_fn({"params": vf_state.params}, t_test, x_test, cond, train=True)
        assert x_out.shape == (10, 5)

        cond_embed = vf.apply({"params": vf_state.params}, cond, method="get_condition_embedding")
        assert cond_embed.shape == (1, 12)
