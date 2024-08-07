import jax
import pytest

from cfp.data.dataloader import TrainSampler


class TestTrainSampler:
    @pytest.mark.parametrize("batch_size", [1, 31])
    def test_sampling_no_combinations(self, pdata, batch_size):
        sampler = TrainSampler(data=pdata, batch_size=batch_size)
        rng_1 = jax.random.PRNGKey(0)
        rng_2 = jax.random.PRNGKey(1)

        sample_1 = sampler.sample(rng_1)
        sample_2 = sampler.sample(rng_2)

        assert "src_cell_data" in sample_1
        assert "tgt_cell_data" in sample_1
        assert "condition" in sample_1
        assert sample_1["src_cell_data"].shape[0] == batch_size
        assert sample_2["src_cell_data"].shape[0] == batch_size
        assert sample_1["tgt_cell_data"].shape[0] == batch_size
        assert sample_2["tgt_cell_data"].shape[0] == batch_size
        assert sample_1["condition"]["dosage"].shape[0] == 1
        assert sample_2["condition"]["dosage"].shape[0] == 1


class TestValidationSampler:
    @pytest.mark.parametrize("n_conditions_on_log_iteration", [None, 1, 3])
    def test_valid_sampler(self, adata_perturbation, n_conditions_on_log_iteration):
        from cfp.data.dataloader import ValidationSampler
        from cfp.data.datamanager import DataManager

        control_key = "control"
        sample_covariates = ["cell_type"]
        sample_covariate_reps = {"cell_type": "cell_type"}

        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key=control_key,
            perturbation_covariates={"drug": ["drug1"]},
            perturbation_covariate_reps={"drug": "drug"},
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
        )

        val_data = dm.get_validation_data(
            adata_perturbation,
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
        )

        s = ValidationSampler(val_data)
        out = s.sample(mode="on_log_iteration")
        assert "source" in out
        assert "target" in out
        assert "condition" in out
        assert len(out["target"]) == len(out["source"])
        assert len(out["target"]) == len(out["condition"])
        assert (
            len(out["target"]) == n_conditions_on_log_iteration
            if n_conditions_on_log_iteration is not None
            else val_data.n_perturbations
        )
        assert out["source"].keys() == out["target"].keys()
        assert out["source"].keys() == out["condition"].keys()
