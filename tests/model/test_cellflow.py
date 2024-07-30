import pytest

import cfp


class TestCellFlow:
    @pytest.mark.parametrize(
        "cell_data",
        ["X", {"obsm": "X_pca"}],
    )
    @pytest.mark.parametrize("solver", ["genot", "otfm"])
    def test_cellflow_cell_data(
        self,
        adata_perturbation,
        cell_data,
        solver,
    ):
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            cell_data=cell_data,
            control_key=("drug1", "control"),
            obs_perturbation_covariates=[("dosage",)],
            uns_perturbation_covariates={"drug": ("drug1",)},
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
    @pytest.mark.parametrize("uns_perturbation_covariates", [{}, {"drug": ("drug1",)}])
    def test_cellflow_uns_covar(
        self,
        adata_perturbation,
        uns_perturbation_covariates,
        solver,
    ):
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            cell_data="X",
            control_key=("drug1", "control"),
            obs_perturbation_covariates=[("dosage",)],
            uns_perturbation_covariates=uns_perturbation_covariates,
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
        "uns_perturbation_covariates",
        [{"drug": ("drug1", "drug2")}, {"drug": "drug1"}],
    )
    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("n_conditions_on_log_iteration", [-1, 0, 3])
    @pytest.mark.parametrize("n_conditions_on_train_end", [-1, 0, 3])
    def test_cellflow_val_data_loading(
        self,
        adata_perturbation,
        split_covariates,
        uns_perturbation_covariates,
        solver,
        n_conditions_on_log_iteration,
        n_conditions_on_train_end,
    ):
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            cell_data="X",
            control_key=("drug1", "control"),
            split_covariates=split_covariates,
            obs_perturbation_covariates=["dosage"],
            uns_perturbation_covariates=uns_perturbation_covariates,
        )
        assert cf.pdata is not None
        assert hasattr(cf, "_data_dim")
        assert cf.pdata.condition_data["drug"].ndim == 3
        assert cf.pdata.condition_data["dosage"].ndim == 3
        assert (
            cf.pdata.condition_data["drug"].shape[1] == cf.pdata.max_combination_length
        )
        assert (
            cf.pdata.condition_data["dosage"].shape[1]
            == cf.pdata.max_combination_length
        )

        cf.prepare_validation_data(
            adata_perturbation,
            name="val",
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )
        assert "val" in cf._validation_data
        assert cf._validation_data["val"].src_data is not None
        assert cf._validation_data["val"].tgt_data is not None
        assert cf._validation_data["val"].condition_data is not None
        assert cf._validation_data["val"].condition_data[0][0]["drug"].ndim == 3
        assert cf._validation_data["val"].condition_data[0][0]["dosage"].ndim == 3
        assert (
            cf._validation_data["val"].max_combination_length
            == cf.pdata.max_combination_length
        )
        assert (
            cf._validation_data["val"].condition_data[0][0]["drug"].shape[1]
            == cf.pdata.max_combination_length
        )
        assert (
            cf._validation_data["val"].condition_data[0][0]["dosage"].shape[1]
            == cf.pdata.max_combination_length
        )
        assert (
            n_conditions_on_log_iteration
            == cf._validation_data["val"].n_conditions_on_log_iteration
        )
        assert (
            n_conditions_on_train_end
            == cf._validation_data["val"].n_conditions_on_train_end
        )

    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("n_conditions_on_log_iteration", [-1, 0, 3])
    @pytest.mark.parametrize("n_conditions_on_train_end", [-1, 0, 3])
    def test_cellflow_with_validation(
        self,
        adata_perturbation,
        solver,
        n_conditions_on_log_iteration,
        n_conditions_on_train_end,
    ):
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            cell_data="X",
            control_key=("drug1", "control"),
            obs_perturbation_covariates=[("dosage",)],
            uns_perturbation_covariates={"drug": ("drug1",)},
        )
        assert cf.pdata is not None
        assert hasattr(cf, "_data_dim")

        cf.prepare_validation_data(
            adata_perturbation,
            name="val",
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )
        assert "val" in cf._validation_data
        assert cf._validation_data["val"].src_data is not None
        assert cf._validation_data["val"].tgt_data is not None
        assert cf._validation_data["val"].condition_data is not None
        assert (
            cf._validation_data["val"].max_combination_length
            == cf.pdata.max_combination_length
        )

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
