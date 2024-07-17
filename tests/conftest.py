import pytest
import jax.numpy as jnp


@pytest.fixture
def dataloader():
    class DataLoader:
        n_conditions = 10

        def sample_batch(self, idx, rng):
            return {
                "src_lin": jnp.ones((10, 5)) * 10,
                "tgt_lin": jnp.ones((10, 5)),
                "src_condition": jnp.ones((10, 2, 3)),
            }

    return DataLoader()
