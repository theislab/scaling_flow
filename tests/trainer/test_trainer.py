import jax
import jax.numpy as jnp
import optax
import pytest
from ott.neural.methods.flows import dynamics, otfm
from ott.solvers import utils as solver_utils

import cfp

x_test = jnp.ones((10, 5)) * 10
t_test = jnp.ones((10, 1))
cond = {"pert1": jnp.ones((1, 2, 3))}
vf_rng = jax.random.PRNGKey(111)
metrics_callback = cfp.training.callbacks.ComputeMetrics(metrics=["r_squared"])


class TestTrainer:
    @pytest.mark.parametrize("valid_freq", [10, 1])
    @pytest.mark.parametrize("callbacks", [[], [metrics_callback]])
    def test_cellflow_trainer(self, dataloader, callbacks, valid_freq):
        opt = optax.adam(1e-3)
        vf = cfp.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        model = otfm.OTFlowMatching(
            vf=vf,
            match_fn=solver_utils.match_linear,
            flow=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt,
            conditions=cond,
            rng=vf_rng,
        )

        trainer = cfp.training.CellFlowTrainer(model=model)
        trainer.train(
            dataloader=dataloader,
            num_iterations=2,
            valid_freq=valid_freq,
            callbacks=callbacks,
        )
        x_pred = trainer.predict(x_test, cond)
        assert x_pred.shape == x_test.shape

        cond_enc = trainer.get_condition_embedding(cond)
        assert cond_enc.shape == (1, 12)
