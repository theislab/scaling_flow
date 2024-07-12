import itertools
from collections.abc import Sequence
from typing import Literal, NamedTuple

import anndata
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

__all__ = []


class PerturbationData(NamedTuple):
    cell_ids: dict[int, str]
    cell_data: jax.Array
    covariate_dict: dict[str, jax.Array]


def filter_adata(adata: anndata.AnnData, filter_dict):
    """
    Filters the AnnData object based on the conditions specified in filter_dict.

    Parameters
    ----------
        adata (anndata.AnnData): The AnnData object to be filtered.
        filter_dict (dict): A dictionary where keys are column names in adata.obs and values are the values to filter by.

    Returns
    -------
        anndata.AnnData: A filtered AnnData object.
    """
    mask = np.ones(len(adata.obs), dtype=bool)

    for column, value in filter_dict.items():
        mask &= adata.obs[column] == value

    return adata[mask]


def load_from_adata(
    adata: anndata.AnnData,
    data: Literal["X"] | dict[str, str],
    source_covariates: Sequence[int],
    control_obs: str,
    conditions_obs: dict[str, str | None],
) -> PerturbationData:
    """Load data from an AnnData object.

    Args:
        adata: An AnnData object.
        data: Where to read the cell data from.
        source_covariates: Column in adata.obs defining the source distribution
        control_obs: Column in adata.obs defining the control condition.
        conditions_obs: Dictionary mapping condition names to observation keys.

    Returns
    -------
        Dictionary of data.
    """

    def get_cell_data(adata):
        if data == "X":
            cell_data = adata.X
            if isinstance(cell_data, sp.csr_matrix):
                cell_data = jnp.asarray(cell_data.toarray())
            else:
                cell_data = jnp.asarray(cell_data)
        else:
            assert isinstance(data, dict)
            assert "obsm" in data
            assert "key" in data
            cell_data = jnp.asarray(adata.obsm[data["obsm"]][data["key"]])
        return cell_data

    src_data = {}
    tgt_data = {}  # this has as keys the values of src_data, and as values further conditions
    d_idx_to_src = {}  # dict of dict mapping source ids to strings
    d_idx_to_tgt = {}  # dict of dict mapping target ids to strings
    covariate_dict = {}
    for covariate in source_covariates:
        assert covariate in adata.obs
        assert adata.obs[covariate].dtype.name == "category"

    src_dist = {covariate: adata.obs[covariate].cat.categories for covariate in source_covariates}
    src_counter = 0
    for src_combination in itertools.product(*src_dist.values()):
        filter_dict = {covariate: value for covariate, value in zip(source_covariates, src_combination, strict=False)}
        adata_filtered = filter_adata(adata, filter_dict)
        if len(adata_filtered) == 0:
            print(f"No cells found for filter {filter_dict}.")
            continue

        adata_filtered_control = adata_filtered[adata_filtered.obs[control_obs]]
        src_data[src_counter] = get_cell_data(adata_filtered_control)
        d_idx_to_src[src_counter] = src_combination
        d_idx_to_tgt[src_counter] = {}

        adata_filtered_target = adata_filtered[~adata_filtered.obs[control_obs]]
        tgt_dist = {covariate: adata.obs[covariate].cat.categories for covariate in conditions_obs.keys()}
        tgt_counter = 0
        tgt_data[src_counter] = {}
        for tgt_combination in itertools.product(*tgt_dist.values()):
            filter_dict_tgt = {
                covariate: value for covariate, value in zip(source_covariates, tgt_combination, strict=False)
            }
            adata_filtered_tmp = filter_adata(adata_filtered_target, filter_dict_tgt)
            if len(adata_filtered_tmp) == 0:
                print(f"No cells found for filter {filter_dict}.")
                continue

            tgt_data[src_counter][tgt_counter]["cell_data"] = get_cell_data(adata_filtered_tmp)
            d_idx_to_tgt[src_counter][tgt_counter] = tgt_combination
            for obs_col in tgt_combination:
                if covariate_dict[obs_col] is None:
                    assert len(adata_filtered.obs[obs_col].values.unique()) == 1
                    tgt_data[src_counter][tgt_counter][f"obs_{obs_col}"] = jnp.asarray(
                        adata_filtered.obs[obs_col].values[0]
                    )
                # if uns_key is not None, take the values from adata.uns[uns_key].values
                # and check that the keys of adata.uns[uns_key] match with the values in
                # adata.obs[obs_key]
                else:
                    uns_key = covariate_dict[obs_col]
                    assert isinstance(adata.uns[uns_key], dict)
                    vals = adata_filtered.obs[obs_col].values
                    assert len(np.unique(vals)) == 1
                    assert vals[0] in adata.uns[uns_key].keys()
                    tgt_data[src_counter][tgt_counter][f"obs_{obs_col}"] = jnp.asarray(adata.uns[uns_key][vals[0]])

            tgt_counter += 1
        src_counter += 1
    return PerturbationData(src_data, tgt_data, d_idx_to_src, d_idx_to_tgt)
