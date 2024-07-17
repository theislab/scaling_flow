import jax
import jax.numpy as jnp
import optax
from ott.neural.methods.flows import dynamics, otfm
from ott.solvers import utils as solver_utils

import cfp

x_test = jnp.ones((10, 5)) * 10
t_test = jnp.ones((10, 1))
cond = jnp.ones((10, 2, 3))
vf_rng = jax.random.PRNGKey(111)


class TestTrainer:
    def test_otfm(self, dataloader):
        opt = optax.adam(1e-3)
        vf = cfp.networks.ConditionalVelocityField(
            output_dim=5,
            condition_encoder="transformer",
            condition_dim=3,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        model = otfm.OTFlowMatching(
            vf=vf,
            match_fn=solver_utils.match_linear,
            flow=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt,
            rng=vf_rng,
        )
        dl_list = [dataloader.sample(vf_rng) for i in range(10)]
        history = model(dl_list, n_iters=2, rng=vf_rng)
        assert isinstance(history, dict)

        x_pred = model.transport(x_test, cond)
        assert x_pred.shape == x_test.shape

    def test_cellflow_trainer(self, dataloader):
        opt = optax.adam(1e-3)
        vf = cfp.networks.ConditionalVelocityField(
            output_dim=5,
            condition_encoder="transformer",
            condition_embedding_dim=12,
            condition_dim=3,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        model = otfm.OTFlowMatching(
            vf=vf,
            match_fn=solver_utils.match_linear,
            flow=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt,
            rng=vf_rng,
        )

        trainer = cfp.training.CellFlowTrainer(model=model)
        trainer.train(dataloader=dataloader, num_iterations=2, valid_freq=1)
        x_pred = trainer.predict(x_test, cond)
        assert x_pred.shape == x_test.shape

        cond_enc = trainer.get_condition_embedding(cond)
        assert cond_enc.shape == (10, 12)
