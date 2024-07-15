import jax
import jax.numpy as jnp
import optax
import pytest

import cfp

x_test = jnp.ones((10, 5))
t_test = jnp.ones((10, 1))
perturbs = jnp.ones((10, 3))
cond = jnp.ones((10, 2, 3))
cond_sizes = jnp.full(cond.shape[0], cond.shape[1])


@pytest.mark.parametrize("condition_encoder", ["transformer", "deepset"])
def test_velocity_field(condition_encoder):
    vf = cfp.networks.ConditionalVelocityField(
        condition_encoder=condition_encoder,
        output_dim=5,
        condition_dim=3,
        hidden_dims=(32, 32),
        output_dims=(32, 32),
    )
    vf_rng = jax.random.PRNGKey(111)
    opt = optax.adam(1e-3)
    vf_state = vf.create_train_state(vf_rng, opt, 5)
    x_out = vf_state.apply_fn({"params": vf_state.params}, t_test, x_test, cond, cond_sizes, training=True)
    assert x_out.shape == (10, 5)
