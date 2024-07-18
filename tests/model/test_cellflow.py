import pytest

import cfp


class TestCellFlow:
    @pytest.mark.parametrize(
        "cell_data",
        ["X", {"obsm": "X_pca"}],
    )
    @pytest.mark.parametrize("uns_perturbation_covariates", [{}, {"drug": ("drug1",)}])
    def test_cellflow(
        self,
        adata_perturbation,
        uns_perturbation_covariates,
        cell_data,
    ):
        cf = cfp.model.cellflow.CellFlow(adata_perturbation, solver="otfm")
        cf.prepare_data(
            cell_data=cell_data,
            control_key=("drug1", "control"),
            obs_perturbation_covariates=[["dosage"]],
            uns_perturbation_covariates=uns_perturbation_covariates,
        )
        assert cf.pdata is not None
        assert hasattr(cf, "_condition_dim")
        assert hasattr(cf, "_data_dim")

        cf.prepare_model(
            condition_encoder="transformer",
            condition_embedding_dim=32,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        assert cf.trainer is not None

        cf.train(num_iterations=1)
        assert cf.dataloader is not None
