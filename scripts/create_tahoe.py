import anndata as ad
import h5py
import zarr
from cellflow.data._utils import write_sharded
from anndata.experimental import read_lazy
from cellflow.data import DataManager
import cupy as cp
import tqdm
import dask
import numpy as np

print("loading data")
with h5py.File("/lustre/groups/ml01/workspace/100mil/100m_int_indices.h5ad", "r") as f:
    adata_all = ad.AnnData(
        obs=ad.io.read_elem(f["obs"]),
        var=read_lazy(f["var"]),
        uns = read_lazy(f["uns"]),
        obsm = read_lazy(f["obsm"]),
    )

dm = DataManager(adata_all,  
    sample_rep="X_pca",
    control_key="control",
    perturbation_covariates={"drugs": ("drug",), "dosage": ("dosage",)},
    perturbation_covariate_reps={"drugs": "drug_embeddings"},
    sample_covariates=["cell_line"],
    sample_covariate_reps={"cell_line": "cell_line_embeddings"},
    split_covariates=["cell_line"],
    max_combination_length=None,
    null_value=0.0
)

cond_data = dm._get_condition_data(adata=adata_all)
cell_data = dm._get_cell_data(adata_all)



n_source_dists = len(cond_data.split_idx_to_covariates)
n_target_dists = len(cond_data.perturbation_idx_to_covariates)

tgt_cell_data = {}
src_cell_data = {}
gpu_per_cov_mask = cp.asarray(cond_data.perturbation_covariates_mask)
gpu_spl_cov_mask = cp.asarray(cond_data.split_covariates_mask)

for src_idx in tqdm.tqdm(range(n_source_dists), desc="Computing source to cell data idcs"):
    mask = gpu_spl_cov_mask == src_idx
    src_cell_data[str(src_idx)] = {
        "cell_data_index": cp.where(mask)[0].get(),
    }

for tgt_idx in tqdm.tqdm(range(n_target_dists), desc="Computing target to cell data idcs"):
    mask = gpu_per_cov_mask == tgt_idx
    tgt_cell_data[str(tgt_idx)] = {
        "cell_data_index": cp.where(mask)[0].get(),
    }



for src_idx in tqdm.tqdm(range(n_source_dists), desc="Computing source to cell data"):
    indices = src_cell_data[str(src_idx)]["cell_data_index"]
    delayed_obj = dask.delayed(lambda x: cell_data[x])(indices)
    src_cell_data[str(src_idx)]["cell_data"] = dask.array.from_delayed(delayed_obj, shape=(len(indices), cell_data.shape[1]), dtype=cell_data.dtype)

for tgt_idx in tqdm.tqdm(range(n_target_dists), desc="Computing target to cell data"):
    indices = tgt_cell_data[str(tgt_idx)]["cell_data_index"]
    delayed_obj = dask.delayed(lambda x: cell_data[x])(indices)
    tgt_cell_data[str(tgt_idx)]["cell_data"] = dask.array.from_delayed(delayed_obj, shape=(len(indices), cell_data.shape[1]), dtype=cell_data.dtype)


split_covariates_mask = np.asarray(cond_data.split_covariates_mask)
perturbation_covariates_mask = np.asarray(cond_data.perturbation_covariates_mask)
condition_data = {str(k): np.asarray(v) for k, v in (cond_data.condition_data or {}).items()}
control_to_perturbation = {str(k): np.asarray(v) for k, v in (cond_data.control_to_perturbation or {}).items()}
split_idx_to_covariates = {str(k): np.asarray(v) for k, v in (cond_data.split_idx_to_covariates or {}).items()}
perturbation_idx_to_covariates = {
    str(k): np.asarray(v) for k, v in (cond_data.perturbation_idx_to_covariates or {}).items()
}
perturbation_idx_to_id = {str(k): v for k, v in (cond_data.perturbation_idx_to_id or {}).items()}

train_data_dict = {
    "split_covariates_mask": split_covariates_mask,
    "perturbation_covariates_mask": perturbation_covariates_mask,
    "split_idx_to_covariates": split_idx_to_covariates,
    "perturbation_idx_to_covariates": perturbation_idx_to_covariates,
    "perturbation_idx_to_id": perturbation_idx_to_id,
    "condition_data": condition_data,
    "control_to_perturbation": control_to_perturbation,
    "max_combination_length": int(cond_data.max_combination_length),
    "src_cell_data": src_cell_data,
    "tgt_cell_data": tgt_cell_data,
}


print("writing data")
path = "/lustre/groups/ml01/workspace/100mil/tahoe.zarr"
zgroup = zarr.open_group(path, mode="w")
chunk_size = 65536
shard_size = chunk_size * 16
write_sharded(
    zgroup,
    train_data_dict,
    chunk_size=chunk_size,
    shard_size=shard_size,
    compressors=None,
)
print("done")