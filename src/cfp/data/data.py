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

from cfp._constants import CONTROL_HELPER
from cfp._types import ArrayLike

from .utils import _to_list

__all__ = ["PerturbationData"]


@dataclass
class PerturbationData:
    """Data container for the perturbation data.

    Parameters
    ----------
    cell_data
        The representation of cell data, e.g. PCA of gene expression data.
    split_covariates_mask
        Mask of the split covariates.
    split_idx_to_covariates
        Dictionary explaining values in split_covariates_mask.
    perturbation_covariates_mask
        Mask of the perturbation covariates.
    perturbation_idx_to_covariates
        Dictionary explaining values in perturbation_covariates_mask.
    condition_data
        Dictionary with embeddings for conditions.
    pert_embedding_idx_to_covariates
        Dictionary explaining values in condition_data.
    control_to_perturbation
        Mapping from control index to target distribution indices.
    max_combination_length
        Maximum number of covariates in a combination.
    """

    cell_data: jax.Array  # (n_cells, n_features)
    split_covariates_mask: (
        jax.Array
    )  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[
        int, str
    ]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_covariates_mask: (
        jax.Array
    )  # (n_cells,), which cell assigned to which target distribution
    perturbation_idx_to_covariates: dict[
        int, tuple[str, ...]
    ]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    condition_data: jax.Array | None  # (n_targets,) all embeddings for conditions
    pert_embedding_idx_to_covariates: dict[
        int, str
    ]  # (n_targets,) dictionary explaining condition_data
    control_to_perturbation: dict[
        int, jax.Array
    ]  # mapping from control idx to target distribution idcs
    max_combination_length: int

    @staticmethod
    def _get_cell_data(
        adata: anndata.AnnData, cell_data: Literal["X"] | dict[str, str]
    ) -> jax.Array:
        error_message = "`cell_data` should be either `X`, a key in `adata.obsm` or a dictionary of the form {`attr`: `key`}."
        if cell_data == "X":
            cell_data = adata.X
            if isinstance(cell_data, sp.csr_matrix):
                return jnp.asarray(cell_data.toarray())
            else:
                return jnp.asarray(cell_data)
        if isinstance(cell_data, str):
            if cell_data not in adata.obsm:
                raise error_message
            return jnp.asarray(adata.obsm[cell_data])
        if not isinstance(cell_data, dict):
            raise error_message
        attr = list(cell_data.keys())[0]
        key = list(cell_data.values())[0]
        return jnp.asarray(getattr(adata, attr)[key])

    @staticmethod
    def _verify_control_data(adata: anndata.AnnData, data: tuple[str, Any]):
        if not isinstance(data, tuple | list):
            raise ValueError(f"Control data should be a tuple of length 2, got {data}.")
        if len(data) != 2:
            raise ValueError(f"Control data should be a tuple of length 2, got {data}.")
        if data[0] not in adata.obs:
            raise ValueError(f"Control column {data[0]} not found in adata.obs.")
        assert data[0] in adata.obs, f"Control column {data[0]} not found in adata.obs."
        if not isinstance(adata.obs[data[0]].dtype, pd.CategoricalDtype):
            try:
                adata.obs[data[0]] = adata.obs[data[0]].astype("category")
            except ValueError:
                raise ValueError(
                    f"Control column {data[0]} could not be converted to categorical."
                ) from None
        if data[1] not in adata.obs[data[0]].cat.categories:
            raise ValueError(f"Control value {data[1]} not found in {data[0]}.")

    @classmethod
    def _verify_obs_perturbation_covariates(
        cls, adata: anndata.AnnData, data: Any
    ) -> None:
        if not isinstance(data, Sequence):
            raise ValueError(
                f"Data should be a sequence, found {data} to be of type {type(data)}."
            )
        for group in data:
            if not isinstance(group, tuple):
                raise ValueError(
                    f"Group should be a tuple, found {group} to be of type {type(group)}."
                )
            cls._verify_covariate_data(adata, group)

    @classmethod
    def _verify_uns_perturbation_covariates(
        cls, adata: anndata.AnnData, data: Any
    ) -> None:
        if not isinstance(data, dict):
            raise ValueError(
                f"Data should be a dictionary, found {data} to be of type {type(data)}."
            )
        for key, group in data.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Key should be a string, found {group} to be of type {type(group)}."
                )
            if not isinstance(group, tuple):
                raise ValueError(
                    f"Group should be a tuple, found {group} to be of type {type(group)}."
                )
            cls._verify_covariate_data(adata, group)

    @staticmethod
    def _verify_covariate_data(adata: anndata.AnnData, group: Any) -> None:
        for covariate in group:
            if covariate not in adata.obs:
                raise ValueError(f"Covariate {covariate} not found in adata.obs.")
            if not isinstance(adata.obs[covariate].dtype, pd.CategoricalDtype):
                try:
                    adata.obs[covariate] = adata.obs[covariate].astype("category")
                except ValueError:
                    raise ValueError(
                        f"Covariate {covariate} could not be converted to categorical."
                    ) from None

    @staticmethod
    def _check_shape(arr: float | ArrayLike) -> ArrayLike:
        if not hasattr(arr, "shape") or len(arr.shape) == 0:
            return np.ones((1, 1)) * arr
        if arr.ndim == 1:  # type: ignore[union-attr]
            return arr[:, None]  # type: ignore[index]
        elif arr.ndim == 2:  # type: ignore[union-attr]
            if arr.shape[0] == 1:
                return arr  # type: ignore[return-value]
            if arr.shape[1] == 1:
                return np.transpose(arr)
            raise ValueError("TODO, wrong shape.")
        elif arr.ndim > 2:  # type: ignore[union-attr]
            raise ValueError("TODO. Too many dimensions.")

        raise ValueError("TODO. wrong data for embedding.")

    @staticmethod
    def _get_pert_emb_idx_to_covariate(
        obs_perturbation_covariates: Any,
        uns_perturbation_covariates: Any,
    ) -> dict[int, str]:
        pert_emb_idx_to_covariate = {}
        counter = 0
        if len(obs_perturbation_covariates):
            for obs_group in obs_perturbation_covariates:
                pert_emb_idx_to_covariate[counter] = obs_group
                counter += 1
        if len(uns_perturbation_covariates):
            for uns_group in uns_perturbation_covariates:
                pert_emb_idx_to_covariate[counter] = uns_group
                counter += 1
        return pert_emb_idx_to_covariate

    @classmethod
    def _get_perturbation_covariates(
        cls,
        adata: anndata.AnnData,
        embedding_dict: dict[str, dict[str, ArrayLike]],
        obs_perturbation_covariates: Any,
        uns_perturbation_covariates: Any,
        max_combination_length: int,
        pert_embedding_idx_to_covariates_reversed: dict[int, str],
    ) -> dict[int, jax.Array]:
        embeddings = {}
        for obs_group in obs_perturbation_covariates:
            obs_group_emb = []
            for obs_col in obs_group:
                values = list(adata.obs[obs_col].unique())
                if len(values) != 1:
                    raise ValueError(
                        f"Expected to find exactly one category within distribution, found {len(values)}."
                    )
                arr = jnp.asarray(adata.obs[obs_col].values[0])
                arr = cls._check_shape(arr)
                obs_group_emb.append(arr)
            if len(obs_group) == 1:
                embeddings[pert_embedding_idx_to_covariates_reversed[obs_group]] = (
                    jnp.tile(obs_group_emb[0], (max_combination_length, 1))
                )
            else:
                embeddings[pert_embedding_idx_to_covariates_reversed[obs_group]] = (
                    jnp.concatenate(obs_group_emb, axis=0)
                )

        for uns_key, uns_group in uns_perturbation_covariates.items():
            uns_group = _to_list(uns_group)
            uns_group_emb = []
            for obs_col in uns_group:
                values = list(adata.obs[obs_col].unique())
                if len(values) != 1:
                    raise ValueError("Too many categories within distribution found")
                if uns_key not in embedding_dict:
                    raise ValueError(f"Key {uns_key} not found in `adata.uns`.")
                if not isinstance(adata.uns[uns_key], dict):
                    raise ValueError(
                        f"Value of key {uns_key} in `adata.uns` should be of type `dict`, found {type(adata.uns[uns_key])}."
                    )
                if values[0] not in embedding_dict[uns_key]:
                    raise ValueError(
                        f"Value {values[0]} not found in `adata.uns[{uns_key}]`."
                    )
                arr = jnp.asarray(embedding_dict[uns_key][values[0]])
                arr = cls._check_shape(arr)
                uns_group_emb.append(arr)
            if len(uns_group) == 1:
                embeddings[pert_embedding_idx_to_covariates_reversed[uns_key]] = (
                    jnp.tile(uns_group_emb[0], (max_combination_length, 1))
                )
            else:
                embeddings[pert_embedding_idx_to_covariates_reversed[uns_key]] = (
                    jnp.concatenate(uns_group_emb, axis=0)
                )

        return embeddings

    @classmethod
    def load_from_adata(
        cls,
        adata: anndata.AnnData,
        cell_data: Literal["X"] | dict[str, str],
        control_data: Sequence[str, Any],
        split_covariates: Sequence[str],
        obs_perturbation_covariates: Sequence[tuple[str, ...]],
        uns_perturbation_covariates: Sequence[dict[str, tuple[str, ...] | str]],
    ) -> "PerturbationData":
        """Load cell data from an AnnData object.

        Args:
            adata: An :class:`~anndata.AnnData` object.
            cell_data: Where to read the cell data from. Either 'X', a key in adata.obsm or a dictionary of the form {attr: key}, where 'attr' is an attribute of the :class:`~anndata.AnnData` object and key is the 'key' in the corresponding key.
            control_data: Tuple of length 2 with first element defining the column in :class:`~anndata.AnnData` and second element defining the value in `adata.obs[control_data[0]]` used to define all control cells.
            split_covariates: Covariates in adata.obs to split all control cells into different control populations. The perturbed cells are also split according to these columns, but if an embedding for these covariates should be encoded in the model, the corresponding column should also be used in `obs_perturbation_covariates` or `uns_perturbation_covariates`.
            obs_perturbation_covariates: Tuples of covariates in adata.obs characterizing the perturbed cells (together with `split_covariates` and `uns_perturbation_covariates`) and encoded by the values as found in `adata.obs`. If a tuple contains more than
            one element, this is interpreted as a combination of covariates that should be treated as an unordered set.
            uns_perturbation_covariates: Dictionaries with keys in adata.uns and values columns in adata.obs which characterize the perturbed cells (together with `split_covariates` and `obs_perturbation_covariates`) and encoded by the values as found in `adata.uns[uns_perturbation_covariates.keys()]`. If a value of the dictionary is a tuple with more than one element, this is interpreted as a combination of covariates that should be treated as an unordered set.

        Returns
        -------
            PerturbationData: Data container for the perturbation data.
        """
        # TODO(@MUCDK): add device to possibly only load to cpu
        if split_covariates is None or len(split_covariates) == 0:
            adata.obs[CONTROL_HELPER] = True
            adata.obs[CONTROL_HELPER] = adata.obs[CONTROL_HELPER].astype("category")
            split_covariates = [CONTROL_HELPER]
        cls._verify_control_data(adata, control_data)
        cls._verify_obs_perturbation_covariates(adata, obs_perturbation_covariates)
        cls._verify_uns_perturbation_covariates(adata, uns_perturbation_covariates)

        obs_combination_length = (
            max(len(comb) for comb in obs_perturbation_covariates)
            if len(obs_perturbation_covariates)
            else 0
        )
        uns_combination_length = (
            max(len(comb) for comb in uns_perturbation_covariates.values())  # type: ignore[attr-defined]
            if len(uns_perturbation_covariates)
            else 0
        )
        max_combination_length = max(obs_combination_length, uns_combination_length)

        for covariate in split_covariates:
            if covariate not in adata.obs:
                raise ValueError(f"Split covariate {covariate} not found in adata.obs.")
            assert adata.obs[covariate].dtype.name == "category"

        pert_embedding_idx_to_covariates = cls._get_pert_emb_idx_to_covariate(
            obs_perturbation_covariates, uns_perturbation_covariates
        )
        pert_embedding_idx_to_covariates_reversed = {
            v: k for k, v in pert_embedding_idx_to_covariates.items()
        }

        src_dist = {
            covariate: adata.obs[covariate].cat.categories
            for covariate in split_covariates
        }
        tgt_dist_obs = {
            covariate: adata.obs[covariate].cat.categories
            for group in obs_perturbation_covariates
            for covariate in group
        }
        tgt_dist_uns = {
            covariate: adata.obs[covariate].cat.categories
            for emb_covariates in uns_perturbation_covariates.values()  # type: ignore[attr-defined]
            for covariate in _to_list(emb_covariates)
        }
        tgt_dist_obs.update(tgt_dist_uns)
        src_counter = 0
        tgt_counter = 0
        src_dists = list(itertools.product(*src_dist.values()))

        control_to_perturbation: dict[int, int] = {}
        cell_data = cls._get_cell_data(adata, cell_data)
        split_covariates_mask = np.full((adata.n_obs,), -1, dtype=jnp.int32)
        split_covariates_to_idx = {}
        perturbation_covariates_mask = np.full((adata.n_obs,), -1, dtype=jnp.int32)
        perturbation_covariates_to_idx = {}
        condition_data: dict[int, list] | None = (
            None
            if (
                len(obs_perturbation_covariates) == 0
                and len(uns_perturbation_covariates) == 0
            )
            else {i: [] for i in range(len(pert_embedding_idx_to_covariates))}
        )

        control_mask = (adata.obs[control_data[0]] == control_data[1]) == 1
        for src_combination in tqdm(src_dists):
            filter_dict = dict(zip(split_covariates, src_combination, strict=False))
            split_cov_mask = (
                adata.obs[list(filter_dict.keys())] == list(filter_dict.values())
            ).all(axis=1)
            mask = split_cov_mask * control_mask
            if mask.sum() == 0:
                continue
            control_to_perturbation[src_counter] = []
            split_covariates_mask[mask] = src_counter
            split_covariates_to_idx[src_counter] = src_combination

            conditional_distributions = []
            for tgt_combination in itertools.product(*tgt_dist_obs.values()):
                mask = (
                    (adata.obs[list(tgt_dist_obs.keys())] == list(tgt_combination)).all(
                        axis=1
                    )
                    * (1 - control_mask)
                    * split_cov_mask
                ) == 1
                if mask.sum() == 0:
                    continue
                conditional_distributions.append(tgt_counter)
                perturbation_covariates_mask[mask] = tgt_counter
                perturbation_covariates_to_idx[tgt_counter] = tgt_combination
                control_to_perturbation[src_counter] = tgt_counter
                if condition_data is not None:
                    embedding = cls._get_perturbation_covariates(
                        adata=adata[mask],
                        embedding_dict=adata.uns,
                        obs_perturbation_covariates=obs_perturbation_covariates,
                        uns_perturbation_covariates=uns_perturbation_covariates,
                        max_combination_length=max_combination_length,
                        pert_embedding_idx_to_covariates_reversed=pert_embedding_idx_to_covariates_reversed,
                    )
                    for pert_cov, emb in embedding.items():
                        condition_data[pert_cov].append(emb)
                tgt_counter += 1
            control_to_perturbation[src_counter] = np.array(conditional_distributions)
            src_counter += 1

        if condition_data is not None:
            for pert_cov, emb in condition_data.items():
                condition_data[pert_cov] = jnp.array(emb)

        return cls(
            cell_data=cell_data,
            split_covariates_mask=jnp.asarray(split_covariates_mask),
            split_idx_to_covariates=split_covariates_to_idx,
            perturbation_covariates_mask=jnp.asarray(perturbation_covariates_mask),
            perturbation_idx_to_covariates=perturbation_covariates_to_idx,
            condition_data=condition_data,
            pert_embedding_idx_to_covariates=pert_embedding_idx_to_covariates,
            control_to_perturbation=control_to_perturbation,
            max_combination_length=max_combination_length,
        )

    @property
    def n_controls(self) -> int:
        """Returns the number of control covariate values."""
        return len(self.split_idx_to_covariates)

    @property
    def n_perturbations(self) -> int:
        """Returns the number of perturbation covariate combinations."""
        return len(self.perturbation_idx_to_covariates)

    @property
    def n_perturbation_covariates(self) -> int:
        """Returns the number of perturbation covariates."""
        return len(self.pert_embedding_idx_to_covariates)

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {
            "n_controls": self.n_controls,
            "n_perturbations": self.n_perturbations,
        }
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"
