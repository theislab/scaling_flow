import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Tuple

import anndata
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from cfp._constants import UNS_KEY_CONDITIONS

__all__ = ["Perturbation_data"]


@dataclass
class PerturbationData:
    """
    Data class for perturbation data used in experiments.

    Attributes:
        control_covariates (Iterable[str]): Covariates defining the control group.
        perturbation_covariates (Iterable[str]): Covariates defining the perturbation group.
        control_covariate_values (Iterable[str]): Values of the control covariates.
        perturbation_covariate_values (Iterable[str]): Values of the perturbation covariates.
        perturbation_covariate_combinations (Iterable[Iterable[str]]): Combinations of perturbation covariates.
        d_idx_to_src (dict[int, str]): Mapping from distribution index to source distribution identifier.
        src_data (dict[int, dict[str, jax.Array]]): Source data indexed by distribution index.
        d_idx_to_tgt (dict[int, str]): Mapping from distribution index to target distribution identifier.
        tgt_data (dict[int, dict[int, dict[str, jax.Array]]]): Target data indexed by distribution and perturbation indices.
    """
    control_covariates: Iterable[str]
    perturbation_covariates: Iterable[str]
    control_covariate_values: Iterable[str]
    perturbation_covariate_values: Iterable[str]
    perturbation_covariate_combinations: Iterable[Iterable[str]]
    d_idx_to_src: dict[int, str]
    src_data: dict[int, dict[str, jax.Array]]
    d_idx_to_tgt: dict[int, str]
    tgt_data: dict[int, dict[int, dict[str, jax.Array]]]

    def __post_init__(self):
        if len(self.perturbation_covariate_combinations) > 0:
            for group in self.perturbation_covariate_combinations:
                assert isinstance(group, list)
                assert len(group) > 1
        self.n_perturbations_given_control = {
            src_dist: len(self.d_idx_to_tgt[src_dist]) for src_dist in self.src_data.keys()
        }
        perturbation_covariate_combs = [el for groups in self.perturbation_covariate_combinations for el in groups]
        self.perturbation_covariate_no_combination = list(
            set(self.perturbation_covariates) - set(perturbation_covariate_combs)
        )
        self.max_length_combination = (
            max([len(group) for group in self.perturbation_covariate_combinations])
            if len(self.perturbation_covariate_combinations) > 0
            else 1
        )

    @property
    def n_controls(self) -> int:
        """Returns the number of control covariate values."""
        return len(self.control_covariate_values)

    @property
    def n_perturbed(self) -> int:
        """Returns the number of perturbation covariate combinations."""
        return len(self.perturbation_covariate_values)

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {"n_controls": self.n_controls, "n_perturbed": self.n_perturbed}
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"


def filter_adata(adata: anndata.AnnData, filter_dict: dict) -> PerturbationData:
    """
    Filters the AnnData object based on the conditions specified in filter_dict.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to be filtered.
    filter_dict : dict
        A dictionary where keys are column names in adata.obs and values are the values to filter by.

    Returns
    -------
    anndata.AnnData
        A filtered AnnData object.
    """
    mask = np.ones(len(adata.obs), dtype=bool)
    for column, value in filter_dict.items():
        mask &= adata.obs[column] == value
    return adata[mask]


def get_cell_data(adata: anndata.AnnData, cell_data: Literal["X"] | dict[str, str]) -> jax.Array:
    """
    Extracts cell data from an AnnData object.

    Parameters
    ----------
    adata 
        The :class:`anndata.AnnData` object containing the cell data.
    cell_data 
        Specification of where to find the cell data. If "X", the data is taken from adata.X.
        If a dictionary, it must contain "attr" and "key" to specify the location.

    Returns
    -------
        The extracted cell data as a JAX array.
    """
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


def _add_perturbation_covariates(adata: anndata.AnnData, adata_filtered: anndata.AnnData, tgt_data: dict[int, dict[int, jax.Array]], obs_col: Tuple[int, int], src_counter: int, tgt_counter: int) -> dict:
    values = list(adata_filtered.obs[obs_col[0]].unique())
    if len(values) != 1:
        raise ValueError("Too many categories within distribution found")

    if obs_col[1] in adata.uns[UNS_KEY_CONDITIONS]:
        assert isinstance(adata.uns[UNS_KEY_CONDITIONS][obs_col[1]], dict)
        vals = adata_filtered.obs[obs_col[0]].values
        assert len(np.unique(vals)) == 1
        assert vals[0] in adata.uns[UNS_KEY_CONDITIONS][obs_col[1]]
        arr = jnp.asarray(adata.uns[UNS_KEY_CONDITIONS][obs_col[1]][vals[0]])
        arr = _check_shape(arr)
        tgt_data[src_counter][tgt_counter][f"{obs_col[0]}"] = arr
        return tgt_data
    if obs_col[1] is None:
        return tgt_data
    else:
        arr = jnp.asarray(adata_filtered.obs[obs_col[1]].values[0])
        arr = _check_shape(arr)
        tgt_data[src_counter][tgt_counter][f"{obs_col[0]}"] = arr
        return tgt_data


def load_from_adata(
    adata: anndata.AnnData,
    cell_data: Literal["X"] | dict[str, str],
    control_covariates: Sequence[str] | None,
    control_data: tuple[str, Any],
    perturbation_covariates: Sequence[
        tuple[str, str | None]
    ],  
    perturbation_covariate_combinations: Sequence[Sequence[str]],  
) -> PerturbationData:
    """Load cell data from an AnnData object.

    Args:
        adata: An :class:`~anndata.AnnData` object.
        cell_data: Where to read the cell data from. If of type :class:`dict`, the key
            "attr" should be present and the value should be an attribute of :class:`~anndata.AnnData`.
            The key `key` should be present and the value should be the key in the respective attribute
        control_covariates: Covariates in adata.obs defining the source distribution.
        control_data: Tuple of length 2 with first element defining the column in :class:`~anndata.AnnData`
          and second element defining the value in `adata.obs[control_data[0]]` used to define the source
          distribution.
        perturbation_covariates: Covariates in adata.obs characterizing the source distribution.
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
    if control_covariates is None or len(control_covariates) == 0:
        adata.obs["control_dummy"] = True
        adata.obs["control_dummy"] = adata.obs["control_dummy"].astype("category")
        control_covariates = ["control_dummy"]
    _verify_control_data(adata, control_data)

    if UNS_KEY_CONDITIONS not in adata.uns:
        adata.uns[UNS_KEY_CONDITIONS] = {}

    src_data = {}
    tgt_data = {}  # this has as keys the values of src_data, and as values further conditions
    d_idx_to_src = {}  # dict of dict mapping source ids to strings
    d_idx_to_tgt = {}  # dict of dict mapping target ids to strings
    for covariate in control_covariates:
        assert covariate in adata.obs
        assert adata.obs[covariate].dtype.name == "category"

    src_dist = {covariate: adata.obs[covariate].cat.categories for covariate in control_covariates}
    src_counter = 0
    src_dists = list(itertools.product(*src_dist.values()))
    for src_combination in tqdm(src_dists):
        filter_dict = {covariate: value for covariate, value in zip(control_covariates, src_combination, strict=False)}
        adata_filtered = filter_adata(adata, filter_dict)
        if len(adata_filtered) == 0:
            continue

        adata_filtered_control = adata_filtered[(adata_filtered.obs[control_data[0]] == control_data[1]).values]
        src_data[src_counter] = {}
        src_data[src_counter]["cell_data"] = get_cell_data(adata_filtered_control, cell_data)
        d_idx_to_src[src_counter] = src_combination
        d_idx_to_tgt[src_counter] = {}

        adata_filtered_target = adata_filtered[~(adata_filtered.obs[control_data[0]] == control_data[1]).values]
        tgt_dist = {covariate[0]: adata.obs[covariate[0]].cat.categories for covariate in perturbation_covariates}
        tgt_counter = 0
        tgt_data[src_counter] = {}
        for tgt_combination in itertools.product(*tgt_dist.values()):
            filter_dict_tgt = {
                covariate[0]: value for covariate, value in zip(perturbation_covariates, tgt_combination, strict=False)
            }
            adata_filtered_tmp = filter_adata(adata_filtered_target, filter_dict_tgt)
            if len(adata_filtered_tmp) == 0:
                continue

            tgt_data[src_counter][tgt_counter] = {}
            tgt_data[src_counter][tgt_counter]["cell_data"] = get_cell_data(adata_filtered_tmp, cell_data)
            d_idx_to_tgt[src_counter][tgt_counter] = tgt_combination
            for obs_col in perturbation_covariates:
                tgt_data = _add_perturbation_covariates(
                    adata, adata_filtered_tmp, tgt_data, obs_col, src_counter, tgt_counter
                )

            tgt_counter += 1
        src_counter += 1

    c_covariate_values = list(d_idx_to_src.values())
    p_covariate_values = [
        perturbation_conf for source_subdict in d_idx_to_tgt.values() for perturbation_conf in source_subdict.values()
    ]

    return PerturbationData(
        control_covariates,
        [p[0] for p in perturbation_covariates],
        c_covariate_values,
        p_covariate_values,
        perturbation_covariate_combinations,
        d_idx_to_src,
        src_data,
        d_idx_to_tgt,
        tgt_data,
    )
