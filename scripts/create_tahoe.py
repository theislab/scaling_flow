from sc_exp_design.models import CellFlow
import anndata as ad
import h5py

from anndata.experimental import read_lazy

print("loading data")
with h5py.File("/lustre/groups/ml01/workspace/100mil/100m_int_indices.h5ad", "r") as f:
    adata_all = ad.AnnData(
        obs=ad.io.read_elem(f["obs"]),
        var=read_lazy(f["var"]),
        uns = read_lazy(f["uns"]),
        obsm = read_lazy(f["obsm"]),
    )
cf = CellFlow()

print(" preparing train data ")
cf.prepare_train_data(adata_all,  
                      sample_rep="X_pca",
        control_key="control",
        perturbation_covariates={"drugs": ("drug",), "dosage": ("dosage",)},
        perturbation_covariate_reps={"drugs": "drug_embeddings"},
        sample_covariates=["cell_line"],
        sample_covariate_reps={"cell_line": "cell_line_embeddings"},
        split_covariates=["cell_line"])




print("writing zarr")
cf.train_data.write_zarr(f"/lustre/groups/ml01/workspace/100mil/tahoe_train_data.zarr")
print("zarr written")
