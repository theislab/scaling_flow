import anndata as ad
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from cfp.data.data import TrainingData
from cfp.data.dataloader import TrainSampler


@pytest.fixture
def dataloader():
    class DataLoader:
        n_conditions = 10

        def sample(self, rng):
            return {
                "src_cell_data": jnp.ones((10, 5)) * 10,
                "tgt_cell_data": jnp.ones((10, 5)),
                "condition": {"pert1": jnp.ones((1, 2, 3))},
            }

    return DataLoader()


@pytest.fixture
def validdata():
    class ValidData:
        n_conditions = 10

        def __init__(self):
            self.src_data = {0: jnp.ones((10, 5)) * 10}
            self.tgt_data = {0: {0: jnp.ones((10, 5))}}
            self.condition_data = {0: {0: {"pert1": jnp.ones((1, 2, 3))}}}

    return {"val": ValidData()}


@pytest.fixture()
def adata_perturbation() -> ad.AnnData:

    n_obs = 500
    n_vars = 50
    n_pca = 10

    X_data = np.random.rand(n_obs, n_vars)

    my_counts = np.random.rand(n_obs, n_vars)

    X_pca = np.random.rand(n_obs, n_pca)

    cell_lines = np.random.choice(["cell_line_a", "cell_line_b", "cell_line_c"], n_obs)
    dosages = np.random.choice([10.0, 100.0, 1000.0], n_obs)
    drugs = ["drug_a", "drug_b", "drug_c"]
    drug1 = np.random.choice(drugs, n_obs)
    drug2 = np.random.choice(drugs, n_obs)
    drug3 = np.random.choice(drugs, n_obs)

    obs_data = pd.DataFrame(
        {
            "cell_type": cell_lines,
            "dosage": dosages,
            "drug1": drug1,
            "drug2": drug2,
            "drug3": drug3,
        }
    )

    # Create an AnnData object
    adata = ad.AnnData(X=X_data, obs=obs_data)

    # Add the random data to .layers and .obsm
    adata.layers["my_counts"] = my_counts
    adata.obsm["X_pca"] = X_pca

    control_idcs = np.random.choice(n_obs, n_obs // 10, replace=False)
    for col in ["drug1", "drug2", "drug3"]:
        adata.obs.loc[[str(idx) for idx in control_idcs], col] = "control"

    for col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype("category")

    drug_emb = {}
    for drug in adata.obs["drug1"].cat.categories:
        drug_emb[drug] = np.random.randn(5, 1)
    adata.uns["drug"] = drug_emb

    cell_type_emb = {}
    for cell_type in adata.obs["cell_type"].cat.categories:
        cell_type_emb[cell_type] = np.random.randn(3, 1)
    adata.uns["cell_type"] = cell_type_emb

    return adata


@pytest.fixture()
def adata_perturbation_with_nulls(adata_perturbation: ad.AnnData) -> ad.AnnData:
    adata = adata_perturbation.copy()
    del adata.obs["drug1"]
    del adata.obs["drug2"]
    del adata.obs["drug3"]
    n_obs = adata.n_obs
    drugs = ["drug_a", "drug_b", "drug_c", "control", "no_drug"]
    drug1 = np.random.choice(drugs, n_obs)
    drug2 = np.random.choice(drugs, n_obs)
    drug3 = np.random.choice(drugs, n_obs)
    adata.obs["drug1"] = drug1
    adata.obs["drug2"] = drug2
    adata.obs["drug3"] = drug3
    adata.obs["drug1"] = adata.obs["drug1"].astype("category")
    adata.obs["drug2"] = adata.obs["drug2"].astype("category")
    adata.obs["drug3"] = adata.obs["drug3"].astype("category")

    return adata


@pytest.fixture()
def pdata(adata_perturbation: ad.AnnData) -> TrainingData:
    cell_data = "X"
    split_covariates = ["cell_type"]
    control_data = ("drug1", "control")
    obs_perturbation_covariates = [("dosage",)]
    uns_perturbation_covariates = {"drug": ("drug1", "drug2")}

    pdata = TrainingData.load_from_adata(
        adata_perturbation,
        cell_data=cell_data,
        split_covariates=split_covariates,
        control_data=control_data,
        obs_perturbation_covariates=obs_perturbation_covariates,
        uns_perturbation_covariates=uns_perturbation_covariates,
    )

    return pdata


@pytest.fixture()
def sampler(pdata: TrainingData):
    return TrainSampler(pdata, batch_size=32)
