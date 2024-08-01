import pytest

import cfp

perturbation_covariate_comb_args = [
    {"drug": ["drug1", "drug2"], "dosage": ["dosage_a", "dosage_b"]},
    {
        "drug": ["drug_a", "drug_b", "drug_c"],
        "dosage": ["dosage_a", "dosage_b", "dosage_c"],
    },
]


class TestCellFlow:
    @pytest.mark.parametrize(
        "sample_rep",
        ["X", {"obsm": "X_pca"}],
    )
    @pytest.mark.parametrize("solver", ["genot", "otfm"])
    def test_cellflow_sample_rep(
        self,
        adata_perturbation,
        sample_rep,
        solver,
    ):
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep=sample_rep,
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
            perturbation_covariate_reps={"drug": "drug"},
        )
        assert cf.pdata is not None
        assert hasattr(cf, "_data_dim")

        condition_encoder_kwargs = {}
        if solver == "genot":
            condition_encoder_kwargs["genot_source_layers"] = (
                ("mlp", {"dims": (32, 32)}),
            )
            condition_encoder_kwargs["genot_source_dim"] = 32

        cf.prepare_model(
            condition_embedding_dim=32,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
            condition_encoder_kwargs=condition_encoder_kwargs,
        )
        assert cf.trainer is not None

        cf.train(num_iterations=2)
        assert cf.dataloader is not None

    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_cellflow_covar_reps(
        self,
        adata_perturbation,
        perturbation_covariate_reps,
        solver,
    ):
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug1", "drug2"]},
            perturbation_covariate_reps=perturbation_covariate_reps,
        )
        assert cf.pdata is not None
        assert hasattr(cf, "_data_dim")

        condition_encoder_kwargs = {}
        if solver == "genot":
            condition_encoder_kwargs["genot_source_layers"] = (
                ("mlp", {"dims": (32, 32)}),
            )
            condition_encoder_kwargs["genot_source_dim"] = 32

        cf.prepare_model(
            condition_embedding_dim=32,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
            condition_encoder_kwargs=condition_encoder_kwargs,
        )
        assert cf.trainer is not None

        cf.train(num_iterations=2)
        assert cf.dataloader is not None

    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize(
        "perturbation_covariates", perturbation_covariate_comb_args
    )
    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("n_conditions_on_log_iteration", [-1, 0, 2])
    @pytest.mark.parametrize("n_conditions_on_train_end", [-1, 0, 2])
    def test_cellflow_val_data_loading(
        self,
        adata_perturbation,
        split_covariates,
        perturbation_covariates,
        solver,
        n_conditions_on_log_iteration,
        n_conditions_on_train_end,
    ):
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep="X",
            control_key="control",
            split_covariates=split_covariates,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps={"drug": "drug"},
        )
        assert cf.pdata is not None
        assert hasattr(cf, "_data_dim")

        for k in perturbation_covariates.keys():
            assert k in cf.pdata.condition_data.keys()
            assert cf.pdata.condition_data[k].ndim == 3
            assert (
                cf.pdata.condition_data[k].shape[1] == cf.pdata.max_combination_length
            )
            assert cf.pdata.condition_data[k].shape[0] == cf.pdata.n_perturbations

        cf.prepare_validation_data(
            adata_perturbation,
            name="val",
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )
        assert isinstance(cf._validation_data, dict)
        assert "val" in cf._validation_data
        assert isinstance(cf._validation_data["val"].src_data, dict)
        assert isinstance(cf._validation_data["val"].tgt_data, dict)
        assert isinstance(cf._validation_data["val"].condition_data, dict)

        cond_data = cf._validation_data["val"].condition_data[0][0]
        assert (
            cf._validation_data["val"].n_conditions_on_log_iteration
            == n_conditions_on_log_iteration
        )
        assert (
            cf._validation_data["val"].n_conditions_on_train_end
            == n_conditions_on_train_end
        )
        for k in perturbation_covariates.keys():
            assert k in cond_data.keys()
            assert cond_data[k].ndim == 3
            assert cond_data[k].shape[1] == cf.pdata.max_combination_length
            assert cond_data[k].shape[0] == 1

    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("n_conditions_on_log_iteration", [-1, 0, 1])
    @pytest.mark.parametrize("n_conditions_on_train_end", [-1, 0, 1])
    def test_cellflow_with_validation(
        self,
        adata_perturbation,
        solver,
        n_conditions_on_log_iteration,
        n_conditions_on_train_end,
    ):
        # TODO(@MUCDK) after PR #33 check for larger n_conditions_on...
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
            perturbation_covariate_reps={"drug": "drug"},
        )
        assert cf.pdata is not None
        assert hasattr(cf, "_data_dim")

        cf.prepare_validation_data(
            adata_perturbation,
            name="val",
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )
        assert isinstance(cf._validation_data, dict)
        assert "val" in cf._validation_data
        assert isinstance(cf._validation_data["val"].src_data, dict)
        assert isinstance(cf._validation_data["val"].tgt_data, dict)
        assert isinstance(cf._validation_data["val"].condition_data, dict)
        assert (
            cf._validation_data["val"].max_combination_length
            == cf.pdata.max_combination_length
        )
        cond_data = cf._validation_data["val"].condition_data[0][0]
        assert "drug" in cond_data.keys()
        assert cond_data["drug"].ndim == 3
        assert cond_data["drug"].shape[1] == cf.pdata.max_combination_length
        assert cond_data["drug"].shape[0] == 1

        condition_encoder_kwargs = {}
        if solver == "genot":
            condition_encoder_kwargs["genot_source_layers"] = (
                ("mlp", {"dims": (32, 32)}),
            )
            condition_encoder_kwargs["genot_source_dim"] = 32

        cf.prepare_model(
            condition_embedding_dim=32,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
            condition_encoder_kwargs=condition_encoder_kwargs,
        )
        assert cf.trainer is not None

        metric_to_compute = "r_squared"
        metrics_callback = cfp.training.callbacks.ComputeMetrics(
            metrics=[metric_to_compute]
        )

        cf.train(num_iterations=2, callbacks=[metrics_callback], valid_freq=1)
        assert cf.dataloader is not None
        assert f"val_{metric_to_compute}" in cf.trainer.training_logs
        assert f"train_{metric_to_compute}" in cf.trainer.training_logs
