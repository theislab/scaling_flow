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


class TestDataManager:
    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariates_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    @pytest.mark.parametrize("sample_covariates", [[], ["dosage_c"]])
    def test_init_DataManager(
        self,
        adata_perturbation: ad.AnnData,
        sample_rep,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
        sample_covariates,
    ):
        from cfp.data.datamanager import DataManager

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
        )
        assert isinstance(dm, DataManager)
        assert dm.sample_rep == sample_rep
        assert dm.control_key == "control"
        assert dm.split_covariates == split_covariates
        assert dm.perturbation_covariates == perturbation_covariates
        assert dm.sample_covariates == sample_covariates

    @pytest.mark.parametrize("el_to_delete", ["drug1", "cell_line_a"])
    def raise_false_uns_dict(self, adata_perturbation: ad.AnnData, el_to_delete):
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

        from cfp.data.datamanager import DataManager

        with pytest.raises(KeyError, match=r"Representation.*not found.*"):
            _ = DataManager(
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
        from cfp.data.datamanager import DataManager

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

        with pytest.raises(
            ValueError, match=r".*perturbation covariate groups must match.*"
        ):
            _ = DataManager(
                adata_perturbation,
                sample_rep=sample_rep,
                split_covariates=split_covariates,
                control_key=control_key,
                perturbation_covariates=perturbation_covariates,
                perturbation_covariate_reps=perturbation_covariate_reps,
            )

    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariates_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    @pytest.mark.parametrize("sample_covariates", [[], ["dosage_c"]])
    def test_get_train_data(
        self,
        adata_perturbation: ad.AnnData,
        sample_rep,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
        sample_covariates,
    ):
        from cfp.data.data import TrainingDataNew
        from cfp.data.datamanager import DataManager

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
        )
        assert isinstance(dm, DataManager)
        assert dm.sample_rep == sample_rep
        assert dm.control_key == "control"
        assert dm.split_covariates == split_covariates
        assert dm.perturbation_covariates == perturbation_covariates
        assert dm.sample_covariates == sample_covariates

        train_data = dm.get_train_data(adata_perturbation)
        assert isinstance(train_data, TrainingDataNew)
        assert isinstance(train_data, TrainingDataNew)
        assert (
            (train_data.perturbation_covariates_mask == -1)
            + (train_data.split_covariates_mask == -1)
        ).all()
        if split_covariates == []:
            assert train_data.n_controls == 1
        if split_covariates == ["cell_type"]:
            assert train_data.n_controls == len(
                adata_perturbation.obs["cell_type"].cat.categories
            )

        assert isinstance(train_data.condition_data, dict)
        assert isinstance(list(train_data.condition_data.values())[0], jax.Array)
        assert train_data.max_combination_length == 1

        if sample_covariates == [] and perturbation_covariates == {"drug": ("drug1",)}:
            assert (
                train_data.n_perturbations
                == (len(adata_perturbation.obs["drug1"].cat.categories) - 1)
                * train_data.n_controls
            )
        assert isinstance(train_data.cell_data, jax.Array)
        assert isinstance(train_data.split_covariates_mask, jax.Array)
        assert isinstance(train_data.split_idx_to_covariates, dict)
        assert isinstance(train_data.perturbation_covariates_mask, jax.Array)
        assert isinstance(train_data.perturbation_idx_to_covariates, dict)
        assert isinstance(train_data.control_to_perturbation, dict)

    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize(
        "perturbation_covariates", perturbation_covariate_comb_args
    )
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_get_train_data_with_combinations(
        self,
        adata_perturbation: ad.AnnData,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
    ):
        from cfp.data.datamanager import DataManager

        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=["cell_type"],
            sample_covariate_reps={"cell_type": "cell_type"},
        )

        train_data = dm.get_train_data(adata_perturbation)

        assert (
            (train_data.perturbation_covariates_mask == -1)
            + (train_data.split_covariates_mask == -1)
        ).all()

        if split_covariates == []:
            assert train_data.n_controls == 1
        if split_covariates == ["cell_type"]:
            assert train_data.n_controls == len(
                adata_perturbation.obs["cell_type"].cat.categories
            )

        assert isinstance(train_data.condition_data, dict)
        assert isinstance(list(train_data.condition_data.values())[0], jax.Array)
        assert train_data.max_combination_length == len(perturbation_covariates["drug"])

        for k in perturbation_covariates.keys():
            assert k in train_data.condition_data.keys()
            assert train_data.condition_data[k].ndim == 3
            assert (
                train_data.condition_data[k].shape[1]
                == train_data.max_combination_length
            )
            assert train_data.condition_data[k].shape[0] == train_data.n_perturbations

        for k, v in perturbation_covariate_reps.items():
            assert k in train_data.condition_data.keys()
            assert (
                train_data.condition_data[v].shape[1]
                == train_data.max_combination_length
            )
            assert train_data.condition_data[v].shape[0] == train_data.n_perturbations
            cov_key = perturbation_covariates[v][0]
            if cov_key == "drug_a":
                cov_name = cov_key
            else:
                cov_name = adata_perturbation.obs[cov_key].values[0]
            assert (
                train_data.condition_data[v].shape[2]
                == adata_perturbation.uns[k][cov_name].shape[0]
            )

        assert isinstance(train_data.cell_data, jax.Array)
        assert isinstance(train_data.split_covariates_mask, jax.Array)
        assert isinstance(train_data.split_idx_to_covariates, dict)
        assert isinstance(train_data.perturbation_covariates_mask, jax.Array)
        assert isinstance(train_data.perturbation_idx_to_covariates, dict)
        assert isinstance(train_data.control_to_perturbation, dict)

    @pytest.mark.parametrize("max_combination_length", [0, 4])
    def test_max_combination_length(self, adata_perturbation, max_combination_length):

        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariates = {"drug": ["drug1"]}
        perturbation_covariate_reps = {"drug": "drug"}

        # if max_combination_length == 0:
        #     with pytest.warns(UserWarning, match=r".*max_combination_length.*"):
        #         pdata = TrainingData.load_from_adata(
        #             adata,
        #             sample_rep=sample_rep,
        #             split_covariates=split_covariates,
        #             control_key=control_key,
        #             perturbation_covariates=perturbation_covariates,
        #             perturbation_covariate_reps=perturbation_covariate_reps,
        #             max_combination_length=max_combination_length,
        #         )
        # else:
        from cfp.data.datamanager import DataManager

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            max_combination_length=max_combination_length,
        )

        train_data = dm.get_train_data(adata_perturbation)

        assert (
            (train_data.perturbation_covariates_mask == -1)
            + (train_data.split_covariates_mask == -1)
        ).all()

        expected_max_combination_length = max(
            max_combination_length, len(perturbation_covariates["drug"])
        )
        assert dm.max_combination_length == expected_max_combination_length
        assert (
            train_data.condition_data["drug"].shape[1]
            == expected_max_combination_length
        )


class TestValidationDataNew:
    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize(
        "perturbation_covariates", perturbation_covariate_comb_args
    )
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_load_from_adata_with_combinations_new(
        self,
        adata_perturbation: ad.AnnData,
        sample_rep,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
    ):
        from cfp.data.datamanager import DataManager

        control_key = "control"
        sample_covariates = ["cell_type"]
        sample_covariate_reps = {"cell_type": "cell_type"}

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
        )

        val_data = dm.get_validation_data(adata_perturbation)

        assert isinstance(val_data.cell_data, jax.Array)
        assert isinstance(val_data.split_covariates_mask, jax.Array)
        assert isinstance(val_data.split_idx_to_covariates, dict)
        assert isinstance(val_data.perturbation_covariates_mask, jax.Array)
        assert isinstance(val_data.perturbation_idx_to_covariates, dict)
        assert isinstance(val_data.control_to_perturbation, dict)
        assert val_data.max_combination_length == len(perturbation_covariates["drug"])

        assert isinstance(val_data.condition_data, dict)
        assert isinstance(list(val_data.condition_data.values())[0], jax.Array)

        if sample_covariates == [] and perturbation_covariates == {"drug": ("drug1",)}:
            assert (
                val_data.n_perturbations
                == (len(adata_perturbation.obs["drug1"].cat.categories) - 1)
                * val_data.n_controls
            )

    @pytest.mark.skip(reason="To discuss: why should it raise an error?")
    def test_raises_wrong_max_combination_length_new(self, adata_perturbation):
        from cfp.data.datamanager import DataManager

        max_combination_length = 3
        adata = adata_perturbation
        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariates = {"drug": ["drug1"]}
        perturbation_covariate_reps = {"drug": "drug"}

        with pytest.raises(
            ValueError,
            match=r".*max_combination_length.*",
        ):
            dm = DataManager(
                adata,
                sample_rep=sample_rep,
                split_covariates=split_covariates,
                control_key=control_key,
                perturbation_covariates=perturbation_covariates,
                perturbation_covariate_reps=perturbation_covariate_reps,
                max_combination_length=max_combination_length,
            )

            _ = dm.get_validation_data(adata_perturbation)


class TestPredictionData:
    @pytest.mark.skip(reason="TODO")
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
        from cfp.data.data import PredictionData, TrainingData

        control_key = "control"
        sample_covariates = ["cell_type"]
        sample_covariate_reps = {"cell_type": "cell_type"}

        train_data = TrainingData.load_from_adata(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
        )

        pred_data = PredictionData.load_from_adata(
            adata_perturbation,
            sample_rep=sample_rep,
            categorical=train_data.categorical,
            covariate_encoder=train_data.covariate_encoder,
            max_combination_length=train_data.max_combination_length,
            split_covariates=split_covariates,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
        )
        assert isinstance(pred_data, PredictionData)
        assert isinstance(pred_data.src_data, dict)
        assert isinstance(pred_data.condition_data, dict)
        assert pred_data.max_combination_length == len(perturbation_covariates["drug"])

        comb_data = pred_data.condition_data[0][0]
        for k in perturbation_covariates.keys():
            assert k in comb_data.keys()
            assert comb_data[k].ndim == 3
            assert comb_data[k].shape[1] == pred_data.max_combination_length
            assert comb_data[k].shape[0] == 1

        for k, v in perturbation_covariate_reps.items():
            assert k in comb_data.keys()
            assert comb_data[v].shape[1] == pred_data.max_combination_length
            assert comb_data[v].shape[0] == 1
            cov_key = perturbation_covariates[v][0]
            if cov_key == "drug_a":
                cov_name = cov_key
            else:
                cov_name = adata_perturbation.obs[cov_key].values[0]
            assert comb_data[v].shape[2] == adata_perturbation.uns[k][cov_name].shape[0]


class TestConditionData:
    @pytest.mark.skip(reason="TODO")
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
        from cfp.data.data import ConditionData, TrainingData

        control_key = "control"
        sample_covariates = ["cell_type"]
        sample_covariate_reps = {"cell_type": "cell_type"}

        train_data = TrainingData.load_from_adata(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
        )

        cond_data = ConditionData.load_from_adata(
            adata_perturbation,
            categorical=train_data.categorical,
            covariate_encoder=train_data.covariate_encoder,
            max_combination_length=train_data.max_combination_length,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
        )
        assert isinstance(cond_data, ConditionData)
        assert isinstance(cond_data.condition_data, dict)
        assert cond_data.max_combination_length == len(perturbation_covariates["drug"])

        comb_data = cond_data.condition_data[0]
        for k in perturbation_covariates.keys():
            assert k in comb_data.keys()
            assert comb_data[k].ndim == 3
            assert comb_data[k].shape[1] == cond_data.max_combination_length
            assert comb_data[k].shape[0] == 1

        for k, v in perturbation_covariate_reps.items():
            assert k in comb_data.keys()
            assert comb_data[v].shape[1] == cond_data.max_combination_length
            assert comb_data[v].shape[0] == 1
            cov_key = perturbation_covariates[v][0]
            if cov_key == "drug_a":
                cov_name = cov_key
            else:
                cov_name = adata_perturbation.obs[cov_key].values[0]
            assert comb_data[v].shape[2] == adata_perturbation.uns[k][cov_name].shape[0]
