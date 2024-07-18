import anndata as ad
import jax
import pytest


class TestPerturbationData:
    @pytest.mark.parametrize(
        "cell_data",
        ["X", "X_pca", {"obsm": "X_pca"}, {"layers": "my_counts"}],
    )
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("control_data", [("drug1", "control")])
    @pytest.mark.parametrize("obs_perturbation_covariates", [[], [["dosage"]]])
    @pytest.mark.parametrize("uns_perturbation_covariates", [{}, {"drug": ("drug1",)}])
    def test_load_from_adata_no_combinations(
        self,
        adata_perturbation: ad.AnnData,
        cell_data,
        split_covariates,
        control_data,
        obs_perturbation_covariates,
        uns_perturbation_covariates,
    ):
        from cfp.data.data import PerturbationData

        pdata = PerturbationData.load_from_adata(
            adata_perturbation,
            cell_data=cell_data,
            split_covariates=split_covariates,
            control_data=control_data,
            obs_perturbation_covariates=obs_perturbation_covariates,
            uns_perturbation_covariates=uns_perturbation_covariates,
        )
        assert isinstance(pdata, PerturbationData)
        if split_covariates == []:
            assert pdata.n_controls == 1
        if split_covariates == ["cell_type"]:
            assert pdata.n_controls == len(
                adata_perturbation.obs["cell_type"].cat.categories
            )
        if obs_perturbation_covariates == [] and uns_perturbation_covariates == {}:
            assert pdata.n_perturbations == pdata.n_controls
            assert pdata.condition_data is None
            assert pdata.max_combination_length == 0
        else:
            assert isinstance(pdata.condition_data, jax.Array)
            assert pdata.max_combination_length == 1

        if (
            obs_perturbation_covariates == ["dosage"]
            and uns_perturbation_covariates == []
        ):
            assert pdata.n_perturbations == len(
                adata_perturbation.obs["dosage"].cat.categories
            )
        if obs_perturbation_covariates == [] and uns_perturbation_covariates == {
            "drug": ("drug1",)
        }:
            assert (
                pdata.n_perturbations
                == (len(adata_perturbation.obs["drug1"].cat.categories) - 1)
                * pdata.n_controls
            )
        assert isinstance(pdata.cell_data, jax.Array)
        assert isinstance(pdata.split_covariates_mask, jax.Array)
        assert isinstance(pdata.split_idx_to_covariates, dict)
        assert isinstance(pdata.perturbation_covariates_mask, jax.Array)
        assert isinstance(pdata.perturbation_idx_to_covariates, dict)
        assert isinstance(pdata.control_to_perturbation, dict)

    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("control_data", [("drug1", "control")])
    @pytest.mark.parametrize("obs_perturbation_covariates", [[], [["dosage"]]])
    @pytest.mark.parametrize(
        "uns_perturbation_covariates", [{"drug": ("drug1", "drug2")}]
    )
    def test_load_from_adata_with_combinations(
        self,
        adata_perturbation: ad.AnnData,
        split_covariates,
        control_data,
        obs_perturbation_covariates,
        uns_perturbation_covariates,
    ):
        from cfp.data.data import PerturbationData

        pdata = PerturbationData.load_from_adata(
            adata_perturbation,
            cell_data="X",
            split_covariates=split_covariates,
            control_data=control_data,
            obs_perturbation_covariates=obs_perturbation_covariates,
            uns_perturbation_covariates=uns_perturbation_covariates,
        )
        assert isinstance(pdata, PerturbationData)
        if split_covariates == []:
            assert pdata.n_controls == 1
        if split_covariates == ["cell_type"]:
            assert pdata.n_controls == len(
                adata_perturbation.obs["cell_type"].cat.categories
            )
        if obs_perturbation_covariates == [] and uns_perturbation_covariates == {}:
            assert pdata.n_perturbations == pdata.n_controls
            assert pdata.condition_data is None
            assert pdata.max_combination_length == 0
        else:
            assert isinstance(pdata.condition_data, jax.Array)
            assert pdata.max_combination_length == 2

        if (
            obs_perturbation_covariates == ["dosage"]
            and uns_perturbation_covariates == []
        ):
            assert pdata.n_perturbations == len(
                adata_perturbation.obs["dosage"].cat.categories
            )
        if obs_perturbation_covariates == [] and uns_perturbation_covariates == {
            "drug": ("drug1",)
        }:
            assert (
                pdata.n_perturbations
                == (len(adata_perturbation.obs["drug1"].cat.categories) - 1)
                * pdata.n_controls
            )
        assert isinstance(pdata.cell_data, jax.Array)
        assert isinstance(pdata.split_covariates_mask, jax.Array)
        assert isinstance(pdata.split_idx_to_covariates, dict)
        assert isinstance(pdata.perturbation_covariates_mask, jax.Array)
        assert isinstance(pdata.perturbation_idx_to_covariates, dict)
        assert isinstance(pdata.control_to_perturbation, dict)
