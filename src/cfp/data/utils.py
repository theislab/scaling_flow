import anndata
import numpy as np
import jax.numpy as jnp
import scipy.sparse as sp

import jax.tree_util as jtu
import jax
from typing import Union, Optional, Sequence, Any, Dict, Callable, Literal, NamedTuple
__all__ = []


class PerturbationData(NamedTuple):
    cell_ids: Dict[int, str]
    cell_data: jax.Array
    covariate_dict: Dict[str, jax.Array]

def load_from_adata(adata: anndata.AnnData, data: Union[Literal["X"], Dict[str, str]], conditions_obs: Dict[str, Optional[str]], conditions_obsm: Sequence[str]) -> Dict[str, Union[Sequence, Any]] -> PerturbationData:
    """Load data from an AnnData object.

    Args:
        adata: An AnnData object.
        data: Where to read the cell data from.
        conditions_obs: Dictionary mapping condition names to observation keys.
        conditions_obsm: Dictionary mapping condition names to obsm keys.

    Returns:
        Dictionary of data.
    """
    covariate_dict = {}
    cell_ids = {idx: cell_id for idx, cell_id in enumerate(adata.obs_names)}
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
        
    for obs_key,uns_key in conditions_obs.items():
        # if uns_key is None, just take the values from adata.obs[obs_key]
        if uns_key is None:
            covariate_dict[f"obs_{obs_key}"] = jnp.asarray(adata.obs[obs_key].values)
        # if uns_key is not None, take the values from adata.uns[uns_key].values
        # and check that the keys of adata.uns[uns_key] match with the values in 
        # adata.obs[obs_key]
        else:
            assert isinstance(adata.uns[uns_key], dict)
            assert np.unique(adata.obs[obs_key].values) == np.unique(adata.uns[uns_key].keys())
            d = jtu.tree_map(lambda x: jnp.asarray(x), adata.uns[uns_key])
            covariate_dict[f"obs_{obs_key}"] = d
    for obsm_key in conditions_obsm:
        covariate_dict[f"obsm_{obsm_key}"] = jnp.asarray(adata.obsm[obsm_key])
    return PerturbationData(cell_ids, cell_data, covariate_dict)