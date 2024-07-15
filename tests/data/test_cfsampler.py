import jax
import pytest

from cfp.data.data import PerturbationData, load_from_adata
from cfp.data.dataloader import CFSampler


class TestSampling:
    @pytest.mark.parametrize(
        "cell_data", ["X", {"attr": "obsm", "key": "X_pca"}, {"attr": "layers", "key": "my_counts"}]
    )
    @pytest.mark.parametrize("control_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("control_data", [("drug1", "Vehicle")])
    @pytest.mark.parametrize(
        "perturbation_covariates", [[("drug1", "drug"), ("dosage", "dosage"), ("cell_type", "cell_type")]]
    )
    def test_sampling_no_combinations(
        self, adata_perturbation, cell_data, control_covariates, control_data, perturbation_covariates
    ):
        batch_size = 31
        pdata = load_from_adata(
            adata_perturbation,
            cell_data=cell_data,
            control_covariates=control_covariates,
            control_data=control_data,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_combinations=[],
        )
        assert isinstance(pdata, PerturbationData)
        if control_covariates == []:
            assert pdata.n_controls == 1
        else:
            assert pdata.n_controls == len(adata_perturbation.obs["cell_type"].cat.categories)
        assert isinstance(pdata.control_covariates, list)
        assert isinstance(pdata.perturbation_covariates, list)
        assert isinstance(pdata.control_covariate_values, list)
        assert isinstance(pdata.perturbation_covariate_values, list)
        assert pdata.max_length_combination == 1

        sampler = CFSampler(pdata, batch_size=batch_size)
        sample = sampler.sample(jax.random.PRNGKey(0))
        assert sample["src_lin"].shape[0] == batch_size
        assert sample["tgt_lin"].shape[0] == batch_size
        assert sample["src_condition"].shape[0] == batch_size
