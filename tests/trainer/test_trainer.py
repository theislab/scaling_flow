import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ott.neural.methods.flows import dynamics

import scaleflow
from scaleflow.solvers import _otfm
from scaleflow.training import CellFlowTrainer, ComputationCallback, Metrics
from scaleflow.utils import match_linear

x_test = jnp.ones((10, 5)) * 10
t_test = jnp.ones((10, 1))
cond = {"pert1": jnp.ones((1, 2, 3))}
vf_rng = jax.random.PRNGKey(111)
callback_rng = jax.random.PRNGKey(222)


class CustomCallback(ComputationCallback):
    def __init__(self, rng):
        super().__init__()
        self.rng = rng
        self.rng_1, self.rng_2, self.rng = jax.random.split(self.rng, 3)

    def on_train_begin(self):
        pass

    def on_log_iteration(self, source_data, validation_data, predicted_data, solver):
        source_array = source_data["val"]["my_naming_of_pert"]
        pred_1 = solver.predict(source_array, cond, rng=self.rng_1)
        pred_2 = solver.predict(source_array, cond, rng=self.rng_2)
        return {"diff": np.sum(abs(pred_1 - pred_2), axis=0)}

    def on_train_end(self, source_data, validation_data, predicted_data, solver):
        source_array = source_data["val"]["my_naming_of_pert"]
        pred_1 = solver.predict(source_array, cond, rng=self.rng_1)
        pred_2 = solver.predict(source_array, cond, rng=self.rng_2)
        return {"diff": np.sum(abs(pred_1 - pred_2), axis=0)}


class TestTrainer:
    @pytest.mark.parametrize("valid_freq", [10, 1])
    def test_scaleflow_trainer(self, dataloader, valid_freq):
        opt = optax.adam(1e-3)
        vf = scaleflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        model = _otfm.OTFlowMatching(
            vf=vf,
            match_fn=match_linear,
            probability_path=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt,
            conditions=cond,
            rng=vf_rng,
        )

        predict_kwargs = {"max_steps": 3, "throw": False}
        trainer = CellFlowTrainer(solver=model, predict_kwargs=predict_kwargs)
        trainer.train(
            dataloader=dataloader,
            num_iterations=2,
            valid_freq=valid_freq,
        )
        x_pred = model.predict(x_test, cond)
        assert x_pred.shape == x_test.shape

        out = model.get_condition_embedding(cond)
        assert isinstance(out, tuple)
        assert isinstance(out[0], np.ndarray)
        assert isinstance(out[1], np.ndarray)
        assert out[0].shape == (1, 12)
        assert out[1].shape == (1, 12)

    @pytest.mark.parametrize("use_validdata", [True, False])
    def test_scaleflow_trainer_with_callback(self, dataloader, valid_loader, use_validdata):
        opt = optax.adam(1e-3)
        vf = scaleflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        model = _otfm.OTFlowMatching(
            vf=vf,
            match_fn=match_linear,
            probability_path=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt,
            conditions=cond,
            rng=vf_rng,
        )

        metric_to_compute = "e_distance"
        metrics_callback = Metrics(metrics=[metric_to_compute])

        predict_kwargs = {"max_steps": 3, "throw": False}
        trainer = CellFlowTrainer(solver=model, predict_kwargs=predict_kwargs)
        trainer.train(
            dataloader=dataloader,
            valid_loaders=valid_loader if use_validdata else None,
            num_iterations=2,
            valid_freq=1,
            callbacks=[metrics_callback],
        )

        assert "loss" in trainer.training_logs
        if use_validdata:
            assert f"val_{metric_to_compute}_mean" in trainer.training_logs

        x_pred = model.predict(x_test, cond)
        assert x_pred.shape == x_test.shape

        out = model.get_condition_embedding(cond)
        assert isinstance(out, tuple)
        assert isinstance(out[0], np.ndarray)
        assert out[0].shape == (1, 12)
        assert isinstance(out[1], np.ndarray)
        assert out[1].shape == (1, 12)

    def test_scaleflow_trainer_with_custom_callback(self, dataloader, valid_loader):
        opt = optax.adam(1e-3)
        vf = scaleflow.networks.ConditionalVelocityField(
            condition_mode="stochastic",
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        solver = _otfm.OTFlowMatching(
            vf=vf,
            match_fn=match_linear,
            probability_path=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt,
            conditions=cond,
            rng=vf_rng,
        )

        custom_callback = CustomCallback(rng=callback_rng)

        predict_kwargs = {"max_steps": 3, "throw": False}
        trainer = CellFlowTrainer(solver=solver, predict_kwargs=predict_kwargs)
        trainer.train(
            dataloader=dataloader,
            valid_loaders=valid_loader,
            num_iterations=2,
            valid_freq=1,
            callbacks=[custom_callback],
        )

        logs = trainer.training_logs
        assert "diff" in logs
        assert isinstance(logs["diff"][0], np.ndarray)
        assert logs["diff"][0].shape == (5,)
        assert 0 < np.mean(logs["diff"][0]) < 10

    def test_predict_kwargs_iter(self, dataloader, valid_loader):
        opt_1 = optax.adam(1e-3)
        opt_2 = optax.adam(1e-3)
        vf_1 = scaleflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        vf_2 = scaleflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        model_1 = _otfm.OTFlowMatching(
            vf=vf_1,
            match_fn=match_linear,
            probability_path=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt_1,
            conditions=cond,
            rng=vf_rng,
        )
        model_2 = _otfm.OTFlowMatching(
            vf=vf_2,
            match_fn=match_linear,
            probability_path=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt_2,
            conditions=cond,
            rng=vf_rng,
        )

        metric_to_compute = "e_distance"
        metrics_callback = scaleflow.training.Metrics(metrics=[metric_to_compute])

        predict_kwargs_1 = {"max_steps": 3, "throw": False}
        predict_kwargs_2 = {"max_steps": 500, "throw": False}

        trainer_1 = scaleflow.training.CellFlowTrainer(solver=model_1, predict_kwargs=predict_kwargs_1)
        trainer_2 = scaleflow.training.CellFlowTrainer(solver=model_2, predict_kwargs=predict_kwargs_2)

        start_1 = time.time()
        trainer_1.train(
            dataloader=dataloader,
            valid_loaders=valid_loader,
            num_iterations=2,
            valid_freq=1,
            callbacks=[metrics_callback],
        )
        end_1 = time.time()
        diff_1 = end_1 - start_1

        start_2 = time.time()
        trainer_2.train(
            dataloader=dataloader,
            valid_loaders=valid_loader,
            num_iterations=2,
            valid_freq=1,
            callbacks=[metrics_callback],
        )
        end_2 = time.time()
        diff_2 = end_2 - start_2

        assert diff_2 - diff_1 > 1
