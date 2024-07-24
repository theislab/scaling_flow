import pytest

import cfp


class TestCellFlow:
    @pytest.mark.parametrize(
        "cell_data",
        ["X", {"obsm": "X_pca"}],
    )
    def test_cellflow_cell_data(
        self,
        adata_perturbation,
        cell_data,
    ):
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver="otfm")
        cf.prepare_data(
            cell_data=cell_data,
            control_key=("drug1", "control"),
            obs_perturbation_covariates=[("dosage",)],
            uns_perturbation_covariates={"drug": ("drug1",)},
        )
        assert cf.pdata is not None
        assert hasattr(cf, "_data_dim")

        cf.prepare_model(
            condition_embedding_dim=32,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        assert cf.trainer is not None

        cf.train(num_iterations=2)
        assert cf.dataloader is not None

    @pytest.mark.parametrize("uns_perturbation_covariates", [{}, {"drug": ("drug1",)}])
    def test_cellflow_uns_covar(
        self,
        adata_perturbation,
        uns_perturbation_covariates,
    ):
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver="otfm")
        cf.prepare_data(
            cell_data="X",
            control_key=("drug1", "control"),
            obs_perturbation_covariates=[("dosage",)],
            uns_perturbation_covariates=uns_perturbation_covariates,
        )
        assert cf.pdata is not None
        assert hasattr(cf, "_data_dim")

        cf.prepare_model(
            condition_embedding_dim=32,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        assert cf.trainer is not None

        cf.train(num_iterations=2)
        assert cf.dataloader is not None

    def test_cellflow_with_validation(self, adata_perturbation):
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver="otfm")
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
        )
        assert "val" in cf._validation_data
        assert cf._validation_data["val"].src_data is not None
        assert cf._validation_data["val"].tgt_data is not None
        assert cf._validation_data["val"].condition_data is not None
        assert (
            cf._validation_data["val"].max_combination_length
            == cf.pdata.max_combination_length
        )

        cf.prepare_model(
            condition_embedding_dim=32,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
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
