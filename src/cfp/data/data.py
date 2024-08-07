from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import anndata
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from tqdm import tqdm

from .utils import _flatten_list, _to_list

__all__ = ["TrainingData", "ValidationData"]


class PredictionData:
    """Data container to perform prediction.

    Parameters
    ----------
    src_data
        Dictionary with data for source cells.
    condition_data
        Dictionary with embeddings for conditions.
    covariate_encoder
        Encoder for the primary covariate.
    categorical
        Whether the primary covariate is categorical.
    max_combination_length
        Maximum number of covariates in a combination.
    null_value
        Token to use for masking `null_value`.
    """

    src_data: dict[int, jnp.ndarray] | None
    condition_data: dict[int | str, jnp.ndarray] | None
    covariate_encoder: preprocessing.OneHotEncoder | None
    categorical: bool
    max_combination_length: int
    null_value: Any

    @staticmethod
    def _verify_split_covariates(
        covariate_data: pd.DataFrame,
        split_covariates: Sequence[str],
        adata: anndata.AnnData | None = None,
    ) -> None:
        for covar in split_covariates:
            if covar not in covariate_data.columns:
                raise ValueError(
                    f"Covariate '{covar}' is required for prediction but was not found in provided data."
                )
            if adata is not None and covar not in adata.obs.columns:
                raise ValueError(f"Split covariate '{covar}' not found in `adata.obs`.")

    @staticmethod
    def _verify_perturb_covar_keys(
        covariate_data: pd.DataFrame, perturb_covar_keys: Sequence[str]
    ) -> None:
        for covar in perturb_covar_keys:
            if covar is not None and covar not in covariate_data.columns:
                raise ValueError(
                    f"Covariate '{covar}' is required for prediction but was not found in provided data."
                )

    @staticmethod
    def _verify_condition_id_key(
        covariate_data: pd.DataFrame, condition_id_key: str | None
    ) -> None:
        if (
            condition_id_key is not None
            and condition_id_key not in covariate_data.columns
        ):
            raise ValueError(
                f"Condition id key '{condition_id_key}' is required for prediction but was not found in provided data."
            )

    @classmethod
    def load_from_adata(
        cls,
        adata: anndata.AnnData,
        sample_rep: str,
        covariate_encoder: preprocessing.OneHotEncoder | None,
        categorical: bool,
        max_combination_length: int,
        covariate_data: pd.DataFrame | None = None,
        condition_id_key: str | None = None,
        perturbation_covariates: dict[str, Sequence[str]] | None = None,
        perturbation_covariate_reps: dict[str, str] | None = None,
        sample_covariates: Sequence[str] | None = None,
        sample_covariate_reps: dict[str, str] | None = None,
        split_covariates: Sequence[str] | None = None,
        null_value: float = 0.0,
    ) -> "PredictionData":
        """Load cell data from an AnnData object.

        Args:
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

        Returns
        -------
            PredictionData: Data container to perform predictions.
        """
        # TODO: add device to possibly only load to cpu
        covariate_data = covariate_data if covariate_data is not None else adata.obs

        perturbation_covariates = {
            k: _to_list(v) for k, v in perturbation_covariates.items()
        }

        linked_perturb_covars = cls._get_linked_perturbation_covariates(
            perturbation_covariates
        )

        sample_covariates = sample_covariates or []
        sample_covariates = _to_list(sample_covariates)

        perturbation_covariate_reps = perturbation_covariate_reps or {}
        cls._verify_perturbation_covariate_reps(
            adata, perturbation_covariate_reps, perturbation_covariates
        )

        sample_covariate_reps = sample_covariate_reps or {}
        sample_cov_groups = {covar: _to_list(covar) for covar in sample_covariates}
        cls._verify_sample_covariate_reps(
            adata, sample_covariate_reps, sample_cov_groups
        )

        covariate_groups = perturbation_covariates | sample_cov_groups
        covariate_reps = perturbation_covariate_reps | sample_covariate_reps

        split_covariates = split_covariates or []

        cls._verify_max_combination_length(
            perturbation_covariates, max_combination_length
        )

        idx_to_covar, covar_to_idx = cls._get_idx_to_covariate(covariate_groups)

        primary_group, primary_covars = next(iter(perturbation_covariates.items()))
        cls._verify_covariate_type(covariate_data, primary_covars, categorical)

        cls._verify_split_covariates(covariate_data, split_covariates, adata)

        if len(split_covariates) > 0:
            split_cov_combs = adata.obs[split_covariates].drop_duplicates().values
        else:
            split_cov_combs = [[]]

        perturb_covar_keys = _flatten_list(perturbation_covariates.values()) + list(
            sample_covariates
        )

        perturb_covar_keys = [k for k in perturb_covar_keys if k is not None]
        cls._verify_perturb_covar_keys(covariate_data, perturb_covar_keys)

        if condition_id_key is not None:
            cls._verify_condition_id_key(covariate_data, condition_id_key)
            select_keys = perturb_covar_keys + [condition_id_key]
        else:
            select_keys = perturb_covar_keys

        src_data: dict[int, jax.Array] = {}

        conditional = (len(perturbation_covariates) > 0) or (len(sample_covariates) > 0)
        condition_data: dict[int | str, dict[int, list]] | None = (
            {} if conditional else None
        )

        src_counter = 0
        for split_combination in split_cov_combs:
            filter_dict = dict(zip(split_covariates, split_combination, strict=False))
            mask = np.array(
                adata.obs[list(filter_dict.keys())] == list(filter_dict.values())
            ).all(axis=1)

            src_data[src_counter] = cls._get_cell_data(adata[mask, :], sample_rep)

            if conditional:
                condition_data[src_counter] = {}

            covariate_data_mask = (
                covariate_data[list(filter_dict.keys())] == list(filter_dict.values())
            ).all(axis=1)

            perturb_covar_df = covariate_data[covariate_data_mask][
                select_keys
            ].drop_duplicates()

            if condition_id_key is not None:
                perturb_covar_df = perturb_covar_df.set_index(condition_id_key)
            else:
                perturb_covar_df = perturb_covar_df.reset_index()

            pbar = tqdm(perturb_covar_df.iterrows(), total=perturb_covar_df.shape[0])
            for cond_id, tgt_cond in pbar:
                tgt_cond = tgt_cond[perturb_covar_keys]

                if conditional:
                    embedding = cls._get_perturbation_covariates(
                        condition_data=tgt_cond,
                        rep_dict=adata.uns,
                        perturb_covariates=perturbation_covariates,
                        sample_covariates=sample_covariates,
                        covariate_reps=covariate_reps,
                        linked_perturb_covars=linked_perturb_covars,
                        primary_encoder=covariate_encoder,
                        primary_is_cat=categorical,
                        max_combination_length=max_combination_length,
                        null_value=null_value,
                    )

                    condition_data[src_counter][cond_id] = {}
                    for pert_cov, emb in embedding.items():
                        condition_data[src_counter][cond_id][pert_cov] = (
                            jnp.expand_dims(emb, 0)
                        )

            src_counter += 1

        return cls(
            src_data=src_data,
            condition_data=condition_data,
            covariate_encoder=covariate_encoder,
            categorical=categorical,
            max_combination_length=max_combination_length,
            null_value=null_value,
        )


class ConditionData:
    """Data container to get condition embedding.

    Parameters
    ----------
    condition_data
        Dictionary with embeddings for conditions.
    covariate_encoder
        Encoder for the primary covariate.
    categorical
        Whether the primary covariate is categorical.
    max_combination_length
        Maximum number of covariates in a combination.
    null_value
        Token to use for masking `null_value`.
    """

    condition_data: dict[int | str, jnp.ndarray] | None
    covariate_encoder: preprocessing.OneHotEncoder | None
    categorical: bool
    max_combination_length: int
    null_value: Any

    @staticmethod
    def _verify_perturb_covar_keys(
        covariate_data: pd.DataFrame, perturb_covar_keys: Sequence[str]
    ) -> None:
        for covar in perturb_covar_keys:
            if covar is not None and covar not in covariate_data.columns:
                raise ValueError(
                    f"Covariate '{covar}' is required for prediction but was not found in provided data."
                )

    @staticmethod
    def _verify_condition_id_key(
        covariate_data: pd.DataFrame, condition_id_key: str | None
    ) -> None:
        if (
            condition_id_key is not None
            and condition_id_key not in covariate_data.columns
        ):
            raise ValueError(
                f"Condition id key '{condition_id_key}' is required for prediction but was not found in provided data."
            )

    @classmethod
    def load_from_adata(
        cls,
        # TODO: this needs adata only to get the covariate reps, so we could add a
        # separate argument for this so one does not have to create a dummy adata object
        adata: anndata.AnnData,
        covariate_encoder: preprocessing.OneHotEncoder | None,
        categorical: bool,
        max_combination_length: int,
        covariate_data: pd.DataFrame | None = None,
        condition_id_key: str | None = None,
        perturbation_covariates: dict[str, Sequence[str]] | None = None,
        perturbation_covariate_reps: dict[str, str] | None = None,
        sample_covariates: Sequence[str] | None = None,
        sample_covariate_reps: dict[str, str] | None = None,
        null_value: float = 0.0,
    ) -> "ConditionData":
        """Load cell data from an AnnData object.

        Args:
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
            null_value: Value to use for padding to `max_combination_length`.

        Returns
        -------
            ConditionData: Data container to get condition embedding.
        """
        # TODO: add device to possibly only load to cpu
        covariate_data = covariate_data if covariate_data is not None else adata.obs

        perturbation_covariates = {
            k: _to_list(v) for k, v in perturbation_covariates.items()
        }

        linked_perturb_covars = cls._get_linked_perturbation_covariates(
            perturbation_covariates
        )

        sample_covariates = sample_covariates or []
        sample_covariates = _to_list(sample_covariates)

        perturbation_covariate_reps = perturbation_covariate_reps or {}
        cls._verify_perturbation_covariate_reps(
            adata, perturbation_covariate_reps, perturbation_covariates
        )

        sample_covariate_reps = sample_covariate_reps or {}
        sample_cov_groups = {covar: _to_list(covar) for covar in sample_covariates}
        cls._verify_sample_covariate_reps(
            adata, sample_covariate_reps, sample_cov_groups
        )

        covariate_groups = perturbation_covariates | sample_cov_groups
        covariate_reps = perturbation_covariate_reps | sample_covariate_reps

        cls._verify_max_combination_length(
            perturbation_covariates, max_combination_length
        )

        idx_to_covar, covar_to_idx = cls._get_idx_to_covariate(covariate_groups)

        primary_group, primary_covars = next(iter(perturbation_covariates.items()))
        cls._verify_covariate_type(covariate_data, primary_covars, categorical)

        perturb_covar_keys = _flatten_list(perturbation_covariates.values()) + list(
            sample_covariates
        )

        perturb_covar_keys = [k for k in perturb_covar_keys if k is not None]
        cls._verify_perturb_covar_keys(covariate_data, perturb_covar_keys)

        if condition_id_key is not None:
            cls._verify_condition_id_key(covariate_data, condition_id_key)
            select_keys = perturb_covar_keys + [condition_id_key]
        else:
            select_keys = perturb_covar_keys

        conditional = (len(perturbation_covariates) > 0) or (len(sample_covariates) > 0)
        condition_data: dict[int | str, dict[int, list]] | None = (
            {} if conditional else None
        )

        perturb_covar_df = covariate_data[select_keys].drop_duplicates()

        if condition_id_key is not None:
            perturb_covar_df = perturb_covar_df.set_index(condition_id_key)
        else:
            perturb_covar_df = perturb_covar_df.reset_index()

        pbar = tqdm(perturb_covar_df.iterrows(), total=perturb_covar_df.shape[0])
        for cond_id, tgt_cond in pbar:
            tgt_cond = tgt_cond[perturb_covar_keys]

            if conditional:
                embedding = cls._get_perturbation_covariates(
                    condition_data=tgt_cond,
                    rep_dict=adata.uns,
                    perturb_covariates=perturbation_covariates,
                    sample_covariates=sample_covariates,
                    covariate_reps=covariate_reps,
                    linked_perturb_covars=linked_perturb_covars,
                    primary_encoder=covariate_encoder,
                    primary_is_cat=categorical,
                    max_combination_length=max_combination_length,
                    null_value=null_value,
                )

                condition_data[cond_id] = {}
                for pert_cov, emb in embedding.items():
                    condition_data[cond_id][pert_cov] = jnp.expand_dims(emb, 0)

        return cls(
            condition_data=condition_data,
            covariate_encoder=covariate_encoder,
            categorical=categorical,
            max_combination_length=max_combination_length,
            null_value=null_value,
        )


@dataclass
class TrainingData:
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
    control_to_perturbation
        Mapping from control index to target distribution indices.
    max_combination_length
        Maximum number of covariates in a combination.
    data_manager
        The data manager
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
    condition_data: dict[
        str | int, jnp.ndarray
    ]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[
        int, jax.Array
    ]  # mapping from control idx to target distribution idcs
    max_combination_length: int
    data_manager: Any

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
        return len(self.condition_data)

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {
            "n_controls": self.n_controls,
            "n_perturbations": self.n_perturbations,
        }
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"


@dataclass
class ValidationData:
    """Data container for the validation data.

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
    control_to_perturbation
        Mapping from control index to target distribution indices.
    max_combination_length
        Maximum number of covariates in a combination.
    data_manager
        The data manager
    n_conditions_on_log_iteration
        Number of conditions to use for computation callbacks at each logged iteration.
        If :obj:`None`, use all conditions.
    n_conditions_on_train_end
        Number of conditions to use for computation callbacks at the end of training.
        If :obj:`None`, use all conditions.
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
    condition_data: dict[
        str | int, jnp.ndarray
    ]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[
        int, jax.Array
    ]  # mapping from control idx to target distribution idcs
    max_combination_length: int
    data_manager: Any
    n_conditions_on_log_iteration: int | None = None
    n_conditions_on_train_end: int | None = None

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
        return len(self.condition_data)


@dataclass
class PredictionDataNew:
    """Data container to perform prediction.

    Parameters
    ----------
    src_data
        Dictionary with data for source cells.
    condition_data
        Dictionary with embeddings for conditions.
    covariate_encoder
        Encoder for the primary covariate.
    max_combination_length
        Maximum number of covariates in a combination.
    null_value
        Token to use for masking `null_value`.
    """

    cell_data: jax.Array  # (n_cells, n_features)
    split_covariates_mask: (
        jax.Array
    )  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[
        int, str
    ]  # (n_sources,) dictionary explaining split_covariates_mask
    condition_data: dict[
        str | int, jnp.ndarray
    ]  # (n_targets,) all embeddings for conditions
    max_combination_length: int
    data_manager: Any
    n_conditions_on_log_iteration: int | None = None
    n_conditions_on_train_end: int | None = None
