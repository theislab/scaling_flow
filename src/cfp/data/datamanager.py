from collections.abc import Sequence
from typing import Any

import anndata
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.preprocessing as preprocessing
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm

from cfp._logging import logger
from cfp._types import ArrayLike
from cfp.data.data import TrainingDataNew

from .utils import _flatten_list, _to_list


class DataManager:
    """Data manager for handling perturbation data.

    Parameters
    ----------
    adata: An :class:`~anndata.AnnData` object.
    covariate_encoder: Encoder for the primary covariate.
    categorical: Whether the primary covariate is categorical.
    max_combination_length: Maximum number of combinations of primary `perturbation_covariates`.
    sample_rep: Key in `adata.obsm` where the sample representation is stored or "X" to use `adata.X`.
    covariate_data: Dataframe with covariates. If `None`, `adata.obs` is used.
    condition_id_key: Key in `adata.obs` that defines the condition id.
    perturbation_covariates: A dictionary where the keys indicate the name of the covariate group and the values are keys in `adata.obs`. The corresponding columns should be either boolean (presence/abscence of the perturbation) or numeric (concentration or magnitude of the perturbation). If multiple groups are provided, the first is interpreted as the primary perturbation and the others as covariates corresponding to these perturbations, e.g. `{"drug":("drugA", "drugB"), "time":("drugA_time", "drugB_time")}`.
    perturbation_covariate_reps: A dictionary where the keys indicate the name of the covariate group and the values are keys in `adata.uns` storing a dictionary with the representation of the covariates. E.g. `{"drug":"drug_embeddings"}` with `adata.uns["drug_embeddings"] = {"drugA": np.array, "drugB": np.array}`.
    sample_covariates: Keys in `adata.obs` indicating sample covatiates to be taken into account for training and prediction, e.g. `["age", "cell_type"]`.
    sample_covariate_reps: A dictionary where the keys indicate the name of the covariate group and the values are keys in `adata.uns` storing a dictionary with the representation of the covariates. E.g. `{"cell_type": "cell_type_embeddings"}` with `adata.uns["cell_type_embeddings"] = {"cell_typeA": np.array, "cell_typeB": np.array}`.
    split_covariates: Covariates in adata.obs to split all control cells into different control populations. The perturbed cells are also split according to these columns, but if these covariates should also be encoded in the model, the corresponding column should also be used in `perturbation_covariates` or `sample_covariates`.
    null_value: Value to use for padding to `max_combination_length`.


    """

    def __init__(
        self,
        adata: anndata.AnnData,
        sample_rep: str | dict[str, str],
        control_key: str,
        perturbation_covariates: dict[str, Sequence[str]] | None = None,
        perturbation_covariate_reps: dict[str, str] | None = None,
        sample_covariates: Sequence[str] | None = None,
        sample_covariate_reps: dict[str, str] | None = None,
        split_covariates: Sequence[str] | None = None,
        max_combination_length: int | None = None,
        null_value: float = 0.0,
    ):
        self.adata = adata
        self.sample_rep = self._verify_sample_rep(sample_rep)
        self.control_key = control_key
        self.perturbation_covariates = self._verify_perturbation_covariates(
            perturbation_covariates
        )
        self.perturbation_covariate_reps = self._verify_perturbation_covariate_reps(
            adata,
            perturbation_covariate_reps,
            self.perturbation_covariates,
        )
        self.sample_covariates = self._verify_sample_covariates(sample_covariates)
        self.sample_covariate_reps = self._verify_sample_covariate_reps(
            adata, sample_covariate_reps, self.sample_covariates
        )
        self.split_covariates = self._verify_split_covariates(split_covariates)
        self.max_combination_length = self._get_max_combination_length(
            self.perturbation_covariates, max_combination_length
        )
        self.null_value = null_value
        self._primary_one_hot_encoder, self._is_categorical = (
            self._get_primary_covar_encoder(
                self.adata,
                self.perturbation_covariates,
                self.perturbation_covariate_reps,
            )
        )
        self.linked_perturb_covars = self._get_linked_perturbation_covariates(
            self.perturbation_covariates
        )
        sample_cov_groups = {covar: _to_list(covar) for covar in sample_covariates}
        covariate_groups = self.perturbation_covariates | sample_cov_groups
        print("perturbation_covariate_reps ", perturbation_covariate_reps)
        print("sample cov reps", sample_covariate_reps)
        self.covariate_reps = (perturbation_covariate_reps or {}) | (
            sample_covariate_reps or {}
        )

        self.idx_to_covar, self.covar_to_idx = self._get_idx_to_covariate(
            covariate_groups
        )
        perturb_covar_keys = _flatten_list(perturbation_covariates.values()) + list(
            sample_covariates
        )
        self.perturb_covar_keys = [k for k in perturb_covar_keys if k is not None]

    def get_train_data(self, adata: anndata.AnnData) -> Any:
        print(self.perturbation_covariates)
        self._verify_covariate_data(
            adata, {covar: _to_list(covar) for covar in self.sample_covariates}
        )
        self._verify_control_data(adata)
        self._verify_covariate_data(adata, _to_list(self.split_covariates))
        perturb_covar_df = (
            adata.obs[self.perturb_covar_keys].drop_duplicates().reset_index()
        )

        control_to_perturbation: dict[int, int] = {}
        split_covariates_mask = np.full((adata.n_obs,), -1, dtype=jnp.int32)
        split_idx_to_covariates = {}
        perturbation_covariates_mask = np.full((adata.n_obs,), -1, dtype=jnp.int32)
        perturbation_idx_to_covariates = {}

        condition_data: dict[int | str, list[jnp.ndarray]] | None = (
            {i: [] for i in self.covar_to_idx.keys()} if self.is_conditional else None
        )

        control_mask = adata.obs[self.control_key]

        src_counter = 0
        tgt_counter = 0

        if len(self.split_covariates) > 0:
            split_cov_combs = adata.obs[self.split_covariates].drop_duplicates().values
        else:
            split_cov_combs = [[]]

        for split_combination in split_cov_combs:
            filter_dict = dict(
                zip(self.split_covariates, split_combination, strict=False)
            )
            split_cov_mask = (
                adata.obs[list(filter_dict.keys())] == list(filter_dict.values())
            ).all(axis=1)
            mask = np.array(control_mask * split_cov_mask)

            split_covariates_mask[mask] = src_counter
            split_idx_to_covariates[src_counter] = split_combination

            conditional_distributions = []

            pbar = tqdm(perturb_covar_df.iterrows(), total=perturb_covar_df.shape[0])
            for _, tgt_cond in pbar:
                tgt_cond = tgt_cond[self.perturb_covar_keys]
                mask = (adata.obs[self.perturb_covar_keys] == tgt_cond.values).all(
                    axis=1
                )
                mask *= (1 - control_mask) * split_cov_mask
                mask = np.array(mask == 1)

                if mask.sum() == 0:
                    continue

                conditional_distributions.append(tgt_counter)
                perturbation_covariates_mask[mask] = tgt_counter
                perturbation_idx_to_covariates[tgt_counter] = tgt_cond.values

                if self.is_conditional:
                    embedding = self._get_perturbation_covariates(
                        condition_data=tgt_cond,
                        rep_dict=adata.uns,
                        perturb_covariates={
                            k: _to_list(v)
                            for k, v in self.perturbation_covariates.items()
                        },
                    )

                    for pert_cov, emb in embedding.items():
                        condition_data[pert_cov].append(emb)

                tgt_counter += 1

            control_to_perturbation[src_counter] = np.array(conditional_distributions)
            src_counter += 1

        if self.is_conditional:
            for pert_cov, emb in condition_data.items():
                condition_data[pert_cov] = jnp.array(emb)

        return TrainingDataNew(
            cell_data=self._get_cell_data(adata),
            split_covariates_mask=jnp.asarray(split_covariates_mask),
            split_idx_to_covariates=split_idx_to_covariates,
            perturbation_covariates_mask=jnp.asarray(perturbation_covariates_mask),
            perturbation_idx_to_covariates=perturbation_idx_to_covariates,
            condition_data=condition_data,
            control_to_perturbation=control_to_perturbation,
            max_combination_length=self.max_combination_length,
            data_manager=self,
        )

    @staticmethod
    def _verify_sample_rep(sample_rep: str | dict[str, str]) -> str | dict[str, str]:
        if not (isinstance(sample_rep, str) or isinstance(sample_rep, dict)):
            raise ValueError(
                f"`sample_rep` should be of type `str` or `dict`, found {sample_rep} to be of type {type(sample_rep)}."
            )
        return sample_rep

    def _get_cell_data(
        self,
        adata: anndata.AnnData,
    ) -> jax.Array:
        if self.sample_rep == "X":
            sample_rep = adata.X
            if isinstance(sample_rep, sp.csr_matrix):
                return jnp.asarray(sample_rep.toarray())
            else:
                return jnp.asarray(sample_rep)
        if isinstance(self.sample_rep, str):
            if self.sample_rep not in adata.obsm:
                raise KeyError(
                    f"Sample representation '{self.sample_rep}' not found in `adata.obsm`."
                )
            return jnp.asarray(adata.obsm[self.sample_rep])
        attr, key = next(iter(sample_rep.items()))
        return jnp.asarray(getattr(adata, attr)[key])

    def _verify_control_data(self, adata: anndata.AnnData):
        if self.control_key not in adata.obs:
            raise ValueError(
                f"Control column '{self.control_key}' not found in adata.obs."
            )
        if not isinstance(adata.obs[self.control_key].dtype, pd.BooleanDtype):
            try:
                adata.obs[self.control_key] = adata.obs[self.control_key].astype(
                    "boolean"
                )
            except ValueError as e:
                raise ValueError(
                    f"Control column '{self.control_key}' could not be converted to boolean."
                ) from e
        if adata.obs[self.control_key].sum() == 0:
            raise ValueError("No control cells found in adata.")

    @staticmethod
    def _verify_perturbation_covariates(
        data: dict[str, Sequence[str]] | None
    ) -> dict[str, Sequence[str]] | None:
        if not isinstance(data, dict):
            raise ValueError(
                f"`perturbation_covariates` should be a dictionary, found {data} to be of type {type(data)}."
            )
        if len(data) == 0:
            raise ValueError("No perturbation covariates provided.")
        for key, covars in data.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Key should be a string, found {key} to be of type {type(key)}."
                )
            if not isinstance(covars, tuple | list):
                raise ValueError(
                    f"Value should be a tuple, found {covars} to be of type {type(covars)}."
                )
            if len(covars) == 0:
                raise ValueError(
                    f"No covariates provided for perturbation group {key}."
                )
        lengths = [len(covs) for covs in data.values()]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Length of perturbation covariate groups must match, found lengths {lengths}."
            )
        return data

    @staticmethod
    def _verify_sample_covariates(data: Sequence[str] | None) -> Sequence[str] | None:
        if not isinstance(data, tuple | list):
            raise ValueError(
                f"`sample_covariates` should be a tuple or list, found {data} to be of type {type(data)}."
            )
        for covar in data:
            if not isinstance(covar, str):
                raise ValueError(
                    f"Key should be a string, found {covar} to be of type {type(covar)}."
                )
        return data

    @staticmethod
    def _verify_split_covariates(data: Sequence[str] | None) -> Sequence[str] | None:
        if data is None:
            return data
        if not isinstance(data, tuple | list):
            raise ValueError(
                f"`split_covariates` should be a tuple or list, found {data} to be of type {type(data)}."
            )
        for covar in data:
            if not isinstance(covar, str):
                raise ValueError(
                    f"Key should be a string, found {covar} to be of type {type(covar)}."
                )
        return data

    @staticmethod
    def _verify_covariate_data(adata: anndata.AnnData, covars) -> None:
        for covariate in covars:
            if covariate is not None and covariate not in adata.obs:
                raise ValueError(f"Covariate {covariate} not found in adata.obs.")

    @staticmethod
    def _get_linked_perturbation_covariates(
        perturb_covariates: dict[str, Sequence[str]]
    ) -> dict[str, dict[Any, Any]]:

        primary_group, primary_covars = next(iter(perturb_covariates.items()))
        linked_perturb_covars: dict[str, dict[Any, Any]] = {
            k: {} for k in primary_covars
        }
        for cov_group, covars in list(perturb_covariates.items())[1:]:
            for primary_cov, linked_cov in zip(primary_covars, covars, strict=False):
                linked_perturb_covars[primary_cov][cov_group] = linked_cov

        return linked_perturb_covars

    @staticmethod
    def _verify_perturbation_covariate_reps(
        adata: anndata.AnnData,
        data: dict[str, str],
        perturbation_covariate_reps: dict[str, Sequence[str]],
    ) -> dict[str, Sequence[str]]:
        if data is None:
            return None
        for key, value in data.items():
            if key not in perturbation_covariate_reps:
                raise ValueError(
                    f"Key '{key}' not found in `perturbation_covariate_reps`."
                )
            if value not in adata.uns:
                raise ValueError(
                    f"Perturbation covariate representation '{value}' not found in `adata.uns`."
                )
            if not isinstance(adata.uns[value], dict):
                raise ValueError(
                    f"Perturbation covariate representation '{value}' in `adata.uns` should be of type `dict`, found {type(adata.uns[value])}."
                )
        return perturbation_covariate_reps

    @staticmethod
    def _verify_sample_covariate_reps(
        adata: anndata.AnnData,
        data: dict[str, str],
        covariates: dict[str, Sequence[str]],
    ) -> dict[str, Sequence[str]]:
        if data is None:
            return None
        for key, value in data.items():
            if key not in covariates:
                raise ValueError(f"Key '{key}' not found in covariates.")
            if value not in adata.uns:
                raise ValueError(
                    f"Sample covariate representation '{value}' not found in `adata.uns`."
                )
            if not isinstance(adata.uns[value], dict):
                raise ValueError(
                    f"Sample covariate representation '{value}' in `adata.uns` should be of type `dict`, found {type(adata.uns[value])}."
                )
        return covariates

    @staticmethod
    def _get_max_combination_length(
        perturbation_covariates: dict[str, Sequence[str]],
        max_combination_length: int | None,
    ) -> int:
        obs_max_combination_length = max(
            len(comb) for comb in perturbation_covariates.values()
        )
        if max_combination_length is None:
            return obs_max_combination_length
        elif max_combination_length < obs_max_combination_length:
            logger.warning(
                f"Provided `max_combination_length` is smaller than the observed maximum combination length of the perturbation covariates. Setting maximum combination length to {obs_max_combination_length}.",
                stacklevel=2,
            )
            return obs_max_combination_length
        else:
            return max_combination_length

    @staticmethod
    def _verify_max_combination_length(
        perturbation_covariates: dict[str, Sequence[str]], max_combination_length: int
    ) -> int:
        obs_max_combination_length = max(
            len(comb) for comb in perturbation_covariates.values()
        )
        if max_combination_length != obs_max_combination_length:
            raise ValueError(
                f"Observed maximum combination length of the perturbation covariates ({obs_max_combination_length}) does not match `max_combination_length` ({max_combination_length}).",
            )

    def _get_primary_covar_encoder(
        self,
        adata: anndata.AnnData,
        perturbation_covariates: dict[str, Sequence[str]],
        perturbation_covariate_reps: dict[str, str],
    ) -> tuple[preprocessing.OneHotEncoder | None, bool]:
        primary_group, primary_covars = next(iter(perturbation_covariates.items()))
        is_categorical = self._check_covariate_type(adata, primary_covars)
        print("is_categorical", is_categorical)
        print("primary group", primary_group)
        print("is_categorical", is_categorical)
        if primary_group in perturbation_covariate_reps:
            return None, is_categorical
        
        if is_categorical:
            encoder = preprocessing.OneHotEncoder(sparse_output=False)
            all_values = np.unique(adata.obs[primary_covars].values.flatten())
            encoder.fit(all_values.reshape(-1, 1))
            return encoder, is_categorical

        encoder = preprocessing.OneHotEncoder(sparse_output=False)
        encoder.fit(np.array(primary_covars).reshape(-1, 1))
        return encoder, is_categorical

    @staticmethod
    def _check_covariate_type(adata: anndata.AnnData, covars: Sequence[str]) -> bool:
        col_is_cat = []
        for covariate in covars:
            if is_numeric_dtype(adata.obs[covariate]):
                col_is_cat.append(False)
                continue
            if adata.obs[covariate].isin(["True", "False", True, False]).all():
                adata.obs[covariate] = adata.obs[covariate].astype(int)
                col_is_cat.append(False)
                continue
            try:
                adata.obs[covariate] = adata.obs[covariate].astype("category")
                col_is_cat.append(True)
            except ValueError as e:
                raise ValueError(
                    f"Perturbation covariates `{covariate}` should be either numeric/boolean or categorical."
                ) from e

        if max(col_is_cat) != min(col_is_cat):
            raise ValueError(
                f"Groups of perturbation covariates `{covariate}` should be either all numeric/boolean or all categorical."
            )

        return max(col_is_cat)

    @staticmethod
    def _verify_covariate_type(
        covariate_data: pd.DataFrame, covars: Sequence[str], categorical: bool
    ) -> None:
        for covariate in covars:
            if is_numeric_dtype(covariate_data[covariate]):
                if categorical:
                    raise ValueError(
                        f"Perturbation covariates `{covariate}` should be categorical, found numeric."
                    )
                continue
            if covariate_data[covariate].isin(["True", "False", True, False]).all():
                if categorical:
                    raise ValueError(
                        f"Perturbation covariates `{covariate}` should be categorical, found boolean."
                    )
                continue
            try:
                covariate_data[covariate] = covariate_data[covariate].astype("category")
            except ValueError as e:
                raise ValueError(
                    f"Perturbation covariates `{covariate}` should be either numeric/boolean or categorical."
                ) from e
            else:
                if not categorical:
                    raise ValueError(
                        f"Perturbation covariates `{covariate}` should be numeric/boolean, found categorical."
                    )

    @staticmethod
    def _check_shape(arr: float | ArrayLike) -> ArrayLike:
        if not hasattr(arr, "shape") or len(arr.shape) == 0:
            return jnp.ones((1, 1)) * arr
        if arr.ndim == 1:  # type: ignore[union-attr]
            return jnp.expand_dims(arr, 0)  # type: ignore[return-value]
        elif arr.ndim == 2:  # type: ignore[union-attr]
            if arr.shape[0] == 1:
                return arr  # type: ignore[return-value]
            if arr.shape[1] == 1:
                return jnp.transpose(arr)
            raise ValueError(
                "Condition representation has an unexpected shape. Should be (1, n_features) or (n_features, )."
            )
        elif arr.ndim > 2:  # type: ignore[union-attr]
            raise ValueError(
                "Condition representation has too many dimensions. Should be 1 or 2."
            )

        raise ValueError(
            "Condition representation as an unexpected format. Expected an array of shape (1, n_features) or (n_features, )."
        )

    @staticmethod
    def _get_idx_to_covariate(
        covariate_groups: dict[str, Sequence[str]]
    ) -> tuple[dict[int, str], dict[str, int]]:
        idx_to_covar = {}
        for idx, cov_group in enumerate(covariate_groups):
            idx_to_covar[idx] = cov_group
        covar_to_idx = {v: k for k, v in idx_to_covar.items()}
        return idx_to_covar, covar_to_idx

    @staticmethod
    def _pad_to_max_length(
        arr: jax.Array, max_combination_length: int, null_value: Any
    ) -> jax.Array:
        if arr.shape[0] < max_combination_length:
            null_arr = jnp.full(
                (max_combination_length - arr.shape[0], arr.shape[1]), null_value
            )
            arr = jnp.concatenate([arr, null_arr], axis=0)
        return arr

    def _get_perturbation_covariates(
        self,
        condition_data: pd.DataFrame,
        rep_dict: dict[str, dict[str, ArrayLike]],
        perturb_covariates: Any,  # TODO: check if we can save as attribtue
    ) -> dict[str, jax.Array]:
        primary_group, primary_covars = next(iter(perturb_covariates.items()))

        perturb_covar_emb: dict[str, list[jax.Array]] = {
            group: [] for group in perturb_covariates
        }
        for primary_cov in primary_covars:
            value = condition_data[primary_cov]

            cov_name = value if self.is_categorical else primary_cov
            print("primary_group", primary_group)
            print("self.covariate_reps", self.covariate_reps)
            if primary_group in self.covariate_reps:
                rep_key = self.covariate_reps[primary_group]
                if cov_name not in rep_dict[rep_key]:
                    raise ValueError(
                        f"Representation for '{cov_name}' not found in `adata.uns['{primary_group}']`."
                    )
                prim_arr = jnp.asarray(rep_dict[rep_key][cov_name])
            else:
                prim_arr = jnp.asarray(
                    self.primary_one_hot_encoder.transform(
                        np.array(cov_name).reshape(-1, 1)
                    )
                )

            if not self.is_categorical:
                prim_arr *= value

            prim_arr = self._check_shape(prim_arr)
            perturb_covar_emb[primary_group].append(prim_arr)

            for linked_covar in self.linked_perturb_covars[primary_cov].items():
                linked_group, linked_cov = list(linked_covar)

                if linked_cov is None:
                    linked_arr = jnp.full((1, 1), self.null_value)
                    linked_arr = self._check_shape(linked_arr)
                    perturb_covar_emb[linked_group].append(linked_arr)
                    continue

                cov_name = condition_data[linked_cov]

                if linked_group in self.covariate_reps:
                    rep_key = self.covariate_reps[linked_group]
                    if cov_name not in rep_dict[rep_key]:
                        raise ValueError(
                            f"Representation for '{cov_name}' not found in `adata.uns['{linked_group}']`."
                        )
                    linked_arr = jnp.asarray(rep_dict[rep_key][cov_name])
                else:
                    linked_arr = jnp.asarray(condition_data[linked_cov])

                linked_arr = self._check_shape(linked_arr)
                perturb_covar_emb[linked_group].append(linked_arr)

        perturb_covar_emb = {
            k: self._pad_to_max_length(
                jnp.concatenate(v, axis=0), self.max_combination_length, self.null_value
            )
            for k, v in perturb_covar_emb.items()
        }

        sample_covar_emb: dict[str, jax.Array] = {}
        for sample_cov in self.sample_covariates:
            value = condition_data[sample_cov]
            if sample_cov in self.covariate_reps:
                rep_key = self.covariate_reps[sample_cov]
                if value not in rep_dict[rep_key]:
                    raise ValueError(
                        f"Representation for '{value}' not found in `adata.uns['{sample_cov}']`."
                    )
                cov_arr = jnp.asarray(rep_dict[sample_cov][value])
            else:
                cov_arr = jnp.asarray(value)

            cov_arr = self._check_shape(cov_arr)
            sample_covar_emb[sample_cov] = jnp.tile(
                cov_arr, (self.max_combination_length, 1)
            )

        return perturb_covar_emb | sample_covar_emb

    @property
    def primary_one_hot_encoder(self) -> preprocessing.OneHotEncoder | None:
        """One-hot encoder for the primary covariate."""
        return self._primary_one_hot_encoder

    @property
    def is_categorical(self) -> bool:
        """Whether the primary covariate is categorical."""
        return self._is_categorical

    @property
    def is_conditional(self) -> bool:
        """Whether the model is conditional."""
        return (len(self.perturbation_covariates) > 0) or (
            len(self.sample_covariates) > 0
        )
