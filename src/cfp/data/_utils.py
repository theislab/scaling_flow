from collections.abc import Iterable, Sequence
from typing import Any

import anndata as ad
import numpy as np
import sklearn.preprocessing as preprocessing

__all__ = ["encode_onehot"]


def _to_list(x: list[Any] | tuple[Any] | Any) -> list[Any] | tuple[Any]:
    """Converts x to a list if it is not already a list or tuple."""
    if isinstance(x, (list | tuple)):
        return x
    return [x]


def _flatten_list(x: Iterable[Iterable[Any]]) -> list[Any]:
    """Flattens a list of lists."""
    return [item for sublist in x for item in sublist]


def encode_onehot(
    adata: ad.AnnData,
    covariate_keys: Sequence[str],
    uns_key: Sequence[str] = "onehot",
    copy: bool = False,
) -> None | ad.AnnData:
    """Encodes covariates `adata.obs` as one-hot vectors and stores them in `adata.uns`.

    Args:
        adata: Annotated data matrix.
        covariate_keys: Keys of the covariates to encode.
        uns_key: Key in `adata.uns` to store the one-hot encodings.
        copy: Return a copy of `adata` instead of updating it in place.

    Returns
    -------
        If `copy` is `True`, returns a new `AnnData` object with the one-hot encodings stored in `adata.uns`. Otherwise, updates `adata` in place.
    """
    adata = adata.copy() if copy else adata

    all_values = np.unique(adata.obs[covariate_keys].values.flatten()).reshape(-1, 1)
    encoder = preprocessing.OneHotEncoder(sparse_output=False)
    encodings = encoder.fit_transform(all_values)

    adata.uns[uns_key] = {}
    for value, encoding in zip(all_values, encodings, strict=False):
        adata.uns[uns_key][value[0]] = encoding

    if copy:
        return adata
