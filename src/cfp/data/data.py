import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import anndata
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from cfp._constants import CONTROL_HELPER, UNS_KEY_CONDITIONS

__all__ = ["PerturbationData"]


@dataclass
class PerturbationData:
    cell_data: jax.Array  # (n_cells, n_features)
    split_covariates_mask: jax.Array  # (n_cells,), which cell assigned to which source distribution
    split_covariates_to_idx: dict[str, int]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_covariates_mask: jax.Array  # (n_cells,), which cell assigned to which target distribution
    perturbation_covariates_to_idx: dict[str, int]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    condition_mask: jax.Array  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[int, jax.Array]  # mapping from control idx to target distribution idcs

    @property
    def n_controls(self) -> int:
        """Returns the number of control covariate values."""
        return len(self.split_covariates_to_idx)

    @property
    def n_perturbed(self) -> int:
        """Returns the number of perturbation covariate combinations."""
        return len(self.perturbation_covariates_to_idx)

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {"n_controls": self.n_controls, "n_perturbed": self.n_perturbed}
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"


def _get_cell_data(adata: anndata.AnnData, cell_data: Literal["X"] | dict[str, str]) -> jax.Array:
    if cell_data == "X":
        cell_data = adata.X
        if isinstance(cell_data, sp.csr_matrix):
            cell_data = jnp.asarray(cell_data.toarray())
        else:
            cell_data = jnp.asarray(cell_data)
    else:
        assert isinstance(cell_data, dict)
        assert "attr" in cell_data
        assert "key" in cell_data
        cell_data = jnp.asarray(getattr(adata, cell_data["attr"])[cell_data["key"]])
    return cell_data


def _verify_control_data(adata: anndata.AnnData, data: tuple[str, Any]):
    assert isinstance(data, tuple)
    assert len(data) == 2
    assert data[0] in adata.obs
    assert isinstance(adata.obs[data[0]].dtype, pd.CategoricalDtype)
    assert data[1] in adata.obs[data[0]].values


def _check_shape(arr: float | np.ndarray) -> np.ndarray:
    if not hasattr(arr, "shape") or len(arr.shape) == 0:
        return np.ones((1, 1)) * arr

    if arr.ndim == 1:
        return arr[:, None]
    elif arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr
        if arr.shape[1] == 1:
            return np.transpose(arr)
        raise ValueError("TODO, wrong shape.")
    elif arr.ndim > 2:
        raise ValueError("TODO. Too many dimensions.")

    raise ValueError("TODO. wrong data for embedding.")


def _get_perturbation_covariates(
    adata: anndata.AnnData,
    embedding_dict: dict[str, dict[str, np.ndarray]],
    perturbation_covariates: Sequence[tuple[str, str | None]],
) -> dict:
    embeddings = []
    for obs_col in perturbation_covariates:
        values = list(adata.obs[obs_col[0]].unique())
        if len(values) != 1:
            raise ValueError("Too many categories within distribution found")

        if obs_col[1] in embedding_dict:
            assert isinstance(adata.uns[UNS_KEY_CONDITIONS][obs_col[1]], dict)
            vals = adata.obs[obs_col[0]].values
            assert len(np.unique(vals)) == 1
            assert vals[0] in embedding_dict[obs_col[1]]
            arr = jnp.asarray(embedding_dict[obs_col[1]][vals[0]])
            arr = _check_shape(arr)
            embeddings.append(arr)
        elif obs_col[1] is None:
            embeddings.append(_check_shape(values[0]))
        else:
            arr = jnp.asarray(adata.obs[obs_col[1]].values[0])
            arr = _check_shape(arr)
            embeddings.append(arr)
    return jnp.concatenate(embeddings, axis=-1)


def load_from_adata(
    adata: anndata.AnnData,
    cell_data: Literal["X"] | dict[str, str],
    control_data: tuple[str, Any],
    split_covariates: Sequence[tuple[str, str | None]],
    perturbation_covariates: Sequence[tuple[str, str | None]],
    perturbation_covariate_combinations: Sequence[Sequence[str]],
) -> PerturbationData:
    """Load cell data from an AnnData object.

    Args:
        adata: An :class:`~anndata.AnnData` object.
        cell_data: Where to read the cell data from. If of type :class:`dict`, the key
            "attr" should be present and the value should be an attribute of :class:`~anndata.AnnData`.
            The key `key` should be present and the value should be the key in the respective attribute
        control_data: Tuple of length 2 with first element defining the column in :class:`~anndata.AnnData`
          and second element defining the value in `adata.obs[control_data[0]]` used to define the source
          distribution.
        split_covariates: Covariates in adata.obs defining the control distribution.
        perturbation_covariates: Covariates in adata.obs characterizing the source distribution (together
          with `split_covariates`).
          First of a tuple is a column name in adata.obs, the second element is the key in
          adata.uns[`UNS_KEY_CONDITION`] if it exists, otherwise the obs column name if the
          embedding of the covariate should directly be read from it.
        perturbation_covariate_combinations: Groups of covariates that should be combined and treated as
          an unordered set.

    Returns
    -------
        PerturbationData: Data container for the perturbation data.
    """
    # TODO(@MUCDK): add device to possibly only load to cpu
    if split_covariates is None or len(split_covariates) == 0:
        adata.obs[CONTROL_HELPER] = True
        adata.obs[CONTROL_HELPER] = adata.obs[CONTROL_HELPER].astype("category")
        split_covariates = [CONTROL_HELPER]
    _verify_control_data(adata, control_data)

    if UNS_KEY_CONDITIONS not in adata.uns:
        adata.uns[UNS_KEY_CONDITIONS] = {}

    for covariate in split_covariates:
        assert covariate in adata.obs
        assert adata.obs[covariate].dtype.name == "category"

    src_dist = {covariate: adata.obs[covariate].cat.categories for covariate in split_covariates}
    src_counter = 0
    tgt_counter = 0
    src_dists = list(itertools.product(*src_dist.values()))

    control_to_perturbation = {}
    cell_data = _get_cell_data(adata, cell_data)
    split_covariates_mask = np.full((adata.n_obs,), -1, dtype=jnp.int32)
    split_covariates_to_idx = {}
    perturbation_covariates_mask = np.full((adata.n_obs,), -1, dtype=jnp.int32)
    perturbation_covariates_to_idx = {}
    condition_mask = []

    control_mask = adata.obs[control_data[0]] == control_data[1]
    for src_combination in tqdm(src_dists):
        filter_dict = dict(zip(split_covariates, src_combination, strict=False))
        mask = (adata.obs[list(filter_dict.keys())] == list(filter_dict.values())).all(axis=1) * control_mask == 1
        if mask.sum() == 0:
            continue
        split_covariates_mask[mask] = src_counter
        split_covariates_to_idx[src_counter] = src_combination

        tgt_dist = {covariate[0]: adata.obs[covariate[0]].cat.categories for covariate in perturbation_covariates if covariate not in split_coav}
        for tgt_combination in itertools.product(*tgt_dist.values()):
            # TODO check whether we have the split covariances included here. don't think we did before
            filter_dict_tgt = {
                covariate[0]: value for covariate, value in zip(perturbation_covariates, tgt_combination, strict=False)
            }
            print(filter_dict_tgt)
            mask = (adata.obs[list(filter_dict_tgt.keys())] == list(filter_dict_tgt.values())).all(axis=1) * (
                1 - control_mask
            ) == 1
            if mask.sum() == 0:
                continue
            perturbation_covariates_mask[mask]=tgt_counter
            perturbation_covariates_to_idx[tgt_counter] = tgt_combination
            control_to_perturbation[src_counter] = tgt_counter
            embedding = _get_perturbation_covariates(
                adata[mask], adata.uns[UNS_KEY_CONDITIONS], perturbation_covariates
            )
            condition_mask.append(embedding)
            tgt_counter += 1
        src_counter += 1
    condition_mask = jnp.array(condition_mask)

    return PerturbationData(
        cell_data=cell_data,
        split_covariates_mask=split_covariates_mask,
        split_covariates_to_idx=split_covariates_to_idx,
        perturbation_covariates_mask=perturbation_covariates_mask,
        perturbation_covariates_to_idx=perturbation_covariates_to_idx,
        condition_mask=condition_mask,
        control_to_perturbation=control_to_perturbation,
    )
