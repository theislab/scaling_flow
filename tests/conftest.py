import anndata as ad
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from cfp._constants import UNS_KEY_CONDITIONS
from cfp.data.data import PerturbationData


@pytest.fixture
def dataloader():
    class DataLoader:
        n_conditions = 10

        def sample(self, rng):
            return {
                "src_lin": jnp.ones((10, 5)) * 10,
                "tgt_lin": jnp.ones((10, 5)),
                "src_condition": jnp.ones((10, 2, 3)),
            }

    return DataLoader()


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

    adata.uns["cell_flow_conditions"] = {}

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
    adata.uns[UNS_KEY_CONDITIONS]["drug"] = drug_emb

    for drug in adata.obs["cell_type"].cat.categories:
        drug_emb[drug] = np.random.randn(3, 1)
    adata.uns[UNS_KEY_CONDITIONS]["cell_type"] = drug_emb

    return adata


@pytest.fixture()
def pdata(adata_perturbation: ad.AnnData) -> PerturbationData:
    cell_data = "X"
    split_covariates = ["cell_type"]
    control_data = ("drug1", "control")
    obs_perturbation_covariates = [("dosage",)]
    uns_perturbation_covariates = {"drug": ("drug1", "drug2")}

    pdata = PerturbationData.load_from_adata(
        adata_perturbation,
        cell_data=cell_data,
        split_covariates=split_covariates,
        control_data=control_data,
        obs_perturbation_covariates=obs_perturbation_covariates,
        uns_perturbation_covariates=uns_perturbation_covariates,
    )

    return pdata
