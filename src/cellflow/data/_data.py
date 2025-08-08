from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np
import zarr
from zarr.codecs import BloscCodec

from cellflow._types import ArrayLike

__all__ = [
    "BaseDataMixin",
    "ConditionData",
    "PredictionData",
    "TrainingData",
    "ValidationData",
    "ZarrTrainingData",
]


@dataclass
class ReturnData:  # TODO: this should rather be a NamedTuple
    split_covariates_mask: np.ndarray | None
    split_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_covariates_mask: np.ndarray | None
    perturbation_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, np.ndarray]
    control_to_perturbation: dict[int, np.ndarray]
    max_combination_length: int


class BaseDataMixin:
    """Base class for data containers."""

    @property
    def n_controls(self) -> int:
        """Returns the number of control covariate values."""
        return len(self.split_idx_to_covariates)  # type: ignore[attr-defined]

    @property
    def n_perturbations(self) -> int:
        """Returns the number of perturbation covariate combinations."""
        return len(self.perturbation_idx_to_covariates)  # type: ignore[attr-defined]

    @property
    def n_perturbation_covariates(self) -> int:
        """Returns the number of perturbation covariates."""
        return len(self.condition_data)  # type: ignore[attr-defined]

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {
            "n_controls": self.n_controls,
            "n_perturbations": self.n_perturbations,
        }
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"


@dataclass
class ConditionData(BaseDataMixin):
    """Data container containing condition embeddings.

    Parameters
    ----------
    condition_data
        Dictionary with embeddings for conditions.
    max_combination_length
        Maximum number of covariates in a combination.
    null_value
        Token to use for masking `null_value`.
    data_manager
        Data manager used to generate the data.
    """

    condition_data: dict[str, np.ndarray]
    max_combination_length: int
    perturbation_idx_to_covariates: dict[int, tuple[str, ...]]
    perturbation_idx_to_id: dict[int, Any]
    null_value: Any
    data_manager: Any


@dataclass
class TrainingData(BaseDataMixin):
    """Training data.

    Parameters
    ----------
    cell_data
        The representation of cell data, e.g. PCA of gene expression data.
    split_covariates_mask
        Mask of the split covariates.
    split_idx_to_covariates
        Dictionary explaining values in ``split_covariates_mask``.
    perturbation_covariates_mask
        Mask of the perturbation covariates.
    perturbation_idx_to_covariates
        Dictionary explaining values in ``perturbation_covariates_mask``.
    condition_data
        Dictionary with embeddings for conditions.
    control_to_perturbation
        Mapping from control index to target distribution indices.
    max_combination_length
        Maximum number of covariates in a combination.
    data_manager
        The data manager
    """

    cell_data: np.ndarray  # (n_cells, n_features)
    split_covariates_mask: np.ndarray  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[int, tuple[Any, ...]]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_covariates_mask: np.ndarray  # (n_cells,), which cell assigned to which target distribution
    perturbation_idx_to_covariates: dict[
        int, tuple[str, ...]
    ]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, np.ndarray]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[int, np.ndarray]  # mapping from control idx to target distribution idcs
    max_combination_length: int
    null_value: Any
    data_manager: Any

    # --- Zarr export helpers -------------------------------------------------
    def to_zarr(
        self,
        path: str,
        *,
        chunk_size: int = 4096,
        shard_size: int = 65536,
        compressors: Any | None = None,
    ) -> None:
        """Write this training data to Zarr v3 with sharded, compressed arrays.

        Parameters
        ----------
        path
            Path to a Zarr group to create or open for writing.
        chunk_size
            Chunk size along the first axis.
        shard_size
            Shard size along the first axis.
        compressors
            Optional list/tuple of Zarr codecs. If ``None``, a sensible default is used.
        """
        if compressors is None:
            compressors = (BloscCodec(cname="lz4", clevel=3),)

        # Convert to numpy-backed containers for serialization
        cell_data = np.asarray(self.cell_data)
        split_covariates_mask = np.asarray(self.split_covariates_mask)
        perturbation_covariates_mask = np.asarray(self.perturbation_covariates_mask)
        condition_data = {str(k): np.asarray(v) for k, v in (self.condition_data or {}).items()}
        control_to_perturbation = {str(k): np.asarray(v) for k, v in (self.control_to_perturbation or {}).items()}
        split_idx_to_covariates = {str(k): np.asarray(v) for k, v in (self.split_idx_to_covariates or {}).items()}
        perturbation_idx_to_covariates = {
            str(k): np.asarray(v) for k, v in (self.perturbation_idx_to_covariates or {}).items()
        }
        perturbation_idx_to_id = {str(k): v for k, v in (self.perturbation_idx_to_id or {}).items()}

        train_data_dict: dict[str, Any] = {
            "cell_data": cell_data,
            "split_covariates_mask": split_covariates_mask,
            "perturbation_covariates_mask": perturbation_covariates_mask,
            "split_idx_to_covariates": split_idx_to_covariates,
            "perturbation_idx_to_covariates": perturbation_idx_to_covariates,
            "perturbation_idx_to_id": perturbation_idx_to_id,
            "condition_data": condition_data,
            "control_to_perturbation": control_to_perturbation,
            "max_combination_length": int(self.max_combination_length),
        }

        # Ensure Zarr v3 write format for sharding
        ad.settings.zarr_write_format = 3

        def _write_sharded_callback(
            func: Any,
            group: Any,
            key: str,
            element: Any,
            dataset_kwargs: dict[str, Any],
            iospec: Any,
        ) -> None:
            # Only shard/chunk along the first dimension
            if getattr(iospec, "encoding_type", None) in {"array"}:
                dataset_kwargs = {
                    "shards": (shard_size,) + tuple(element.shape[1:]),
                    "chunks": (chunk_size,) + tuple(element.shape[1:]),
                    "compressors": compressors,
                    **dataset_kwargs,
                }
            elif getattr(iospec, "encoding_type", None) in {"csr_matrix", "csc_matrix"}:
                dataset_kwargs = {
                    "shards": (shard_size,),
                    "chunks": (chunk_size,),
                    "compressors": compressors,
                    **dataset_kwargs,
                }

            func(group, key, element, dataset_kwargs=dataset_kwargs)

        zgroup = zarr.open_group(path, mode="a")
        ad.experimental.write_dispatched(zgroup, "/", train_data_dict, callback=_write_sharded_callback)
        zarr.consolidate_metadata(zgroup.store)


@dataclass
class ValidationData(BaseDataMixin):
    """Data container for the validation data.

    Parameters
    ----------
    cell_data
        The representation of cell data, e.g. PCA of gene expression data.
    split_covariates_mask
        Mask of the split covariates.
    split_idx_to_covariates
        Dictionary explaining values in ``split_covariates_mask``.
    perturbation_covariates_mask
        Mask of the perturbation covariates.
    perturbation_idx_to_covariates
        Dictionary explaining values in ``perturbation_covariates_mask``.
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

    cell_data: np.ndarray  # (n_cells, n_features)
    split_covariates_mask: np.ndarray  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[int, tuple[Any, ...]]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_covariates_mask: np.ndarray  # (n_cells,), which cell assigned to which target distribution
    perturbation_idx_to_covariates: dict[
        int, tuple[str, ...]
    ]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, np.ndarray]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[int, np.ndarray]  # mapping from control idx to target distribution idcs
    max_combination_length: int
    null_value: Any
    data_manager: Any
    n_conditions_on_log_iteration: int | None = None
    n_conditions_on_train_end: int | None = None


@dataclass
class PredictionData(BaseDataMixin):
    """Data container to perform prediction.

    Parameters
    ----------
    src_data
        Dictionary with data for source cells.
    condition_data
        Dictionary with embeddings for conditions.
    control_to_perturbation
        Mapping from control index to target distribution indices.
    covariate_encoder
        Encoder for the primary covariate.
    max_combination_length
        Maximum number of covariates in a combination.
    null_value
        Token to use for masking ``null_value``.
    """

    cell_data: ArrayLike  # (n_cells, n_features)
    split_covariates_mask: ArrayLike  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[int, tuple[Any, ...]]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_idx_to_covariates: dict[
        int, tuple[str, ...]
    ]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, ArrayLike]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[int, ArrayLike]
    max_combination_length: int
    null_value: Any
    data_manager: Any


@dataclass
class ZarrTrainingData(BaseDataMixin):
    """Lazy, Zarr-backed variant of :class:`TrainingData`.

    Fields mirror those in :class:`TrainingData`, but array-like members are
    Zarr arrays or Zarr-backed mappings. This enables out-of-core training and
    composition without loading everything into memory.

    Use :meth:`read_zarr` to construct from a Zarr v3 group written via
    :meth:`TrainingData.to_zarr`.
    """

    # Note: annotations use Any to allow zarr.Array and zarr groups without
    # importing zarr at module import time.
    cell_data: Any
    split_covariates_mask: Any
    perturbation_covariates_mask: Any
    split_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_idx_to_covariates: dict[int, tuple[str, ...]]
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, Any]
    control_to_perturbation: dict[int, Any]
    max_combination_length: int

    @classmethod
    def read_zarr(cls, path: str) -> ZarrTrainingData:
        group = zarr.open_group(path, mode="r")
        max_len_node = group.get("max_combination_length")
        if max_len_node is None:
            max_combination_length = 0
        else:
            try:
                max_combination_length = int(max_len_node[()])
            except Exception:  # noqa: BLE001
                max_combination_length = int(max_len_node)

        return cls(
            cell_data=group["cell_data"],
            split_covariates_mask=group["split_covariates_mask"],
            perturbation_covariates_mask=group["perturbation_covariates_mask"],
            split_idx_to_covariates=ad.io.read_elem(group["split_idx_to_covariates"]),
            perturbation_idx_to_covariates=ad.io.read_elem(group["perturbation_idx_to_covariates"]),
            perturbation_idx_to_id=ad.io.read_elem(group["perturbation_idx_to_id"]),
            condition_data=ad.io.read_elem(group["condition_data"]),
            control_to_perturbation=ad.io.read_elem(group["control_to_perturbation"]),
            max_combination_length=max_combination_length,
        )
