import anndata as ad
import jax
import pytest

perturbation_covariates_args = [
    {"drug": ["drug1"]},
    {"drug": ["drug1"], "dosage": ["dosage_a"]},
    {
        "drug": ["drug_a"],
        "dosage": ["dosage_a"],
    },
]

perturbation_covariate_comb_args = [
    {"drug": ["drug1", "drug2"]},
    {"drug": ["drug1", "drug2"], "dosage": ["dosage_a", "dosage_b"]},
    {
        "drug": ["drug_a", "drug_b", "drug_c"],
        "dosage": ["dosage_a", "dosage_b", "dosage_c"],
    },
]


class TestTrainingData:
    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariates_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    @pytest.mark.parametrize("sample_covariates", [[], ["dosage_c"]])
    def test_load_from_adata_no_combinations(
        self,
        adata_perturbation: ad.AnnData,
        sample_rep,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
        sample_covariates,
    ):
        from cfp.data.data import TrainingData

        pdata = TrainingData.load_from_adata(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
        )
        assert isinstance(pdata, TrainingData)
        assert (
            (pdata.perturbation_covariates_mask == -1)
            + (pdata.split_covariates_mask == -1)
        ).all()
        if split_covariates == []:
            assert pdata.n_controls == 1
        if split_covariates == ["cell_type"]:
            assert pdata.n_controls == len(
                adata_perturbation.obs["cell_type"].cat.categories
            )

        assert isinstance(pdata.condition_data, dict)
        assert isinstance(list(pdata.condition_data.values())[0], jax.Array)
        assert pdata.max_combination_length == 1

        if sample_covariates == [] and perturbation_covariates == {"drug": ("drug1",)}:
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
    @pytest.mark.parametrize(
        "perturbation_covariates", perturbation_covariate_comb_args
    )
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_load_from_adata_with_combinations(
        self,
        adata_perturbation: ad.AnnData,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
    ):
        from cfp.data.data import TrainingData

        pdata = TrainingData.load_from_adata(
            adata_perturbation,
            sample_rep="X",
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=["cell_type"],
            sample_covariate_reps={"cell_type": "cell_type"},
        )
        assert isinstance(pdata, TrainingData)

        assert (
            (pdata.perturbation_covariates_mask == -1)
            + (pdata.split_covariates_mask == -1)
        ).all()

        if split_covariates == []:
            assert pdata.n_controls == 1
        if split_covariates == ["cell_type"]:
            assert pdata.n_controls == len(
                adata_perturbation.obs["cell_type"].cat.categories
            )

        assert isinstance(pdata.condition_data, dict)
        assert isinstance(list(pdata.condition_data.values())[0], jax.Array)
        assert pdata.max_combination_length == len(perturbation_covariates["drug"])

        for k in perturbation_covariates.keys():
            assert k in pdata.condition_data.keys()
            assert pdata.condition_data[k].ndim == 3
            assert pdata.condition_data[k].shape[1] == pdata.max_combination_length
            assert pdata.condition_data[k].shape[0] == pdata.n_perturbations

        for k, v in perturbation_covariate_reps.items():
            assert k in pdata.condition_data.keys()
            assert pdata.condition_data[v].shape[1] == pdata.max_combination_length
            assert pdata.condition_data[v].shape[0] == pdata.n_perturbations
            cov_key = perturbation_covariates[v][0]
            if cov_key == "drug_a":
                cov_name = cov_key
            else:
                cov_name = adata_perturbation.obs[cov_key].values[0]
            assert (
                pdata.condition_data[v].shape[2]
                == adata_perturbation.uns[k][cov_name].shape[0]
            )

        assert isinstance(pdata.cell_data, jax.Array)
        assert isinstance(pdata.split_covariates_mask, jax.Array)
        assert isinstance(pdata.split_idx_to_covariates, dict)
        assert isinstance(pdata.perturbation_covariates_mask, jax.Array)
        assert isinstance(pdata.perturbation_idx_to_covariates, dict)
        assert isinstance(pdata.control_to_perturbation, dict)

    @pytest.mark.parametrize("el_to_delete", ["drug1", "cell_line_a"])
    def raise_wrong_uns_dict(self, adata_perturbation: ad.AnnData, el_to_delete):
        from cfp.data.data import TrainingData

        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = ("drug1", "control")
        perturbation_covariates = {"drug": ("drug1", "drug2")}
        perturbation_covariate_reps = {"drug": "drug"}
        sample_covariates = ["cell_type"]
        sample_covariate_reps = {"cell_type": "cell_type"}

        if el_to_delete == "drug1":
            del adata_perturbation.uns["drug"]["drug1"]
        if el_to_delete == "cell_line_a":
            del adata_perturbation.uns["cell_type"]["cell_line_a"]

        with pytest.raises(KeyError):
            _ = TrainingData.load_from_adata(
                adata_perturbation,
                sample_rep=sample_rep,
                split_covariates=split_covariates,
                control_key=control_key,
                perturbation_covariates=perturbation_covariates,
                perturbation_covariate_reps=perturbation_covariate_reps,
                sample_covariates=sample_covariates,
                sample_covariate_reps=sample_covariate_reps,
            )

    @pytest.mark.parametrize("el_to_delete", ["drug1", "dosage_a"])
    def raise_covar_mismatch(self, adata_perturbation: ad.AnnData, el_to_delete):
        from cfp.data.data import TrainingData

        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariate_reps = {"drug": "drug"}
        perturbation_covariates = {
            "drug": ["drug_a", "drug_b"],
            "dosage": ["dosage_a", "dosage_b"],
        }
        if el_to_delete == "drug1":
            perturbation_covariates["drug"] = ["drug_b"]
        if el_to_delete == "cell_line_a":
            perturbation_covariates["dosage"] = ["dosage_b"]

        with pytest.raises(ValueError):
            _ = TrainingData.load_from_adata(
                adata_perturbation,
                sample_rep=sample_rep,
                split_covariates=split_covariates,
                control_key=control_key,
                perturbation_covariates=perturbation_covariates,
                perturbation_covariate_reps=perturbation_covariate_reps,
            )

    @pytest.mark.parametrize("max_combination_length", [0, 4])
    def test_max_combination_length(self, adata_perturbation, max_combination_length):
        from cfp.data.data import TrainingData

        adata = adata_perturbation
        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariates = {"drug": ["drug1"]}
        perturbation_covariate_reps = {"drug": "drug"}

        if max_combination_length == 0:
            with pytest.warns(UserWarning):
                pdata = TrainingData.load_from_adata(
                    adata,
                    sample_rep=sample_rep,
                    split_covariates=split_covariates,
                    control_key=control_key,
                    perturbation_covariates=perturbation_covariates,
                    perturbation_covariate_reps=perturbation_covariate_reps,
                    max_combination_length=max_combination_length,
                )
        else:
            pdata = TrainingData.load_from_adata(
                adata,
                sample_rep=sample_rep,
                split_covariates=split_covariates,
                control_key=control_key,
                perturbation_covariates=perturbation_covariates,
                perturbation_covariate_reps=perturbation_covariate_reps,
                max_combination_length=max_combination_length,
            )

        assert (
            (pdata.perturbation_covariates_mask == -1)
            + (pdata.split_covariates_mask == -1)
        ).all()

        expected_max_combination_length = max(
            max_combination_length, len(perturbation_covariates["drug"])
        )
        assert pdata.max_combination_length == expected_max_combination_length
        assert pdata.condition_data["drug"].shape[1] == expected_max_combination_length


class TestValidationData:
    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize(
        "perturbation_covariates", perturbation_covariate_comb_args
    )
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_load_from_adata_with_combinations(
        self,
        adata_perturbation: ad.AnnData,
        sample_rep,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
    ):
        from cfp.data.data import ValidationData

        val_data = ValidationData.load_from_adata(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=["cell_type"],
            sample_covariate_reps={"cell_type": "cell_type"},
        )
        assert isinstance(val_data, ValidationData)
        assert isinstance(val_data.tgt_data, dict)
        assert isinstance(val_data.src_data, dict)
        assert isinstance(val_data.condition_data, dict)
        assert val_data.max_combination_length == len(perturbation_covariates["drug"])

        comb_data = val_data.condition_data[0][0]
        for k in perturbation_covariates.keys():
            assert k in comb_data.keys()
            assert comb_data[k].ndim == 3
            assert comb_data[k].shape[1] == val_data.max_combination_length
            assert comb_data[k].shape[0] == 1

        for k, v in perturbation_covariate_reps.items():
            assert k in comb_data.keys()
            assert comb_data[v].shape[1] == val_data.max_combination_length
            assert comb_data[v].shape[0] == 1
            cov_key = perturbation_covariates[v][0]
            if cov_key == "drug_a":
                cov_name = cov_key
            else:
                cov_name = adata_perturbation.obs[cov_key].values[0]
            assert comb_data[v].shape[2] == adata_perturbation.uns[k][cov_name].shape[0]

    @pytest.mark.parametrize("max_combination_length", [0, 4])
    def test_max_combination_length(self, adata_perturbation, max_combination_length):
        from cfp.data.data import ValidationData

        adata = adata_perturbation
        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariates = {"drug": ["drug1"]}
        perturbation_covariate_reps = {"drug": "drug"}

        if max_combination_length == 0:
            with pytest.warns(UserWarning):
                pdata = ValidationData.load_from_adata(
                    adata,
                    sample_rep=sample_rep,
                    split_covariates=split_covariates,
                    control_key=control_key,
                    perturbation_covariates=perturbation_covariates,
                    perturbation_covariate_reps=perturbation_covariate_reps,
                    max_combination_length=max_combination_length,
                )
        else:
            pdata = ValidationData.load_from_adata(
                adata,
                sample_rep=sample_rep,
                split_covariates=split_covariates,
                control_key=control_key,
                perturbation_covariates=perturbation_covariates,
                perturbation_covariate_reps=perturbation_covariate_reps,
                max_combination_length=max_combination_length,
            )

        expected_max_combination_length = max(
            max_combination_length, len(perturbation_covariates["drug"])
        )
        assert pdata.max_combination_length == expected_max_combination_length
        assert (
            pdata.condition_data[0][0]["drug"].shape[1]
            == expected_max_combination_length
        )
