from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import anndata as ad
import sklearn.preprocessing as preprocessing

from cfp._logging import logger
from cfp.data._utils import _to_list

__all__ = ["encode_onehot", "annotate_compounds", "get_molecular_fingerprints"]


def annotate_compounds(
    adata,
    query_id: str,
    query_id_type: Literal["name", "cid"] = "name",
    copy: bool = False,
):
    """Annotates compounds in `adata` using pertpy and PubChem.

    Args:
        An :class:`~anndata.AnnData` object.
        query_id: Key in `adata.obs` containing the compound identifiers.
        query_id_type: Type of the compound identifier.
        copy: Return a copy of `adata` instead of updating it in place.

    Returns
    -------
        If `copy` is `True`, returns a new `AnnData` object with the compound annotations stored in `adata.obs`. Otherwise, updates `adata` in place.

        Sets the following fields:
        `.obs["pubchem_name"]`: Name of the compound.
        `.obs["pubchem_ID"]`: PubChem CID of the compound.
        `.obs["smiles"]`: SMILES representation of the compound.
    """
    try:
        import pertpy as pt
    except ImportError:
        raise ImportError(
            "pertpy is not installed. To annotate compounds, please install it via `pip install pertpy`."
        ) from None

    adata = adata.copy() if copy else adata

    c_meta = pt.metadata.Compound()
    c_meta.annotate_compounds(
        adata, query_id=query_id, query_id_type=query_id_type, verbosity=0, copy=False
    )

    not_found = adata.obs[query_id][adata.obs["smiles"].isna()].unique().tolist()
    if not_found:
        logger.warning(
            f"Could not find annotations for the following compounds: {', '.join(not_found)}"
        )

    if copy:
        return adata


def _get_fingerprint(smiles: str, radius: int = 4, n_bits: int = 1024):
    """Computes Morgan fingerprints for a given SMILES string."""
    try:
        from rdkit import Chem
        import rdkit.Chem.rdFingerprintGenerator as rfg
    except ImportError:
        raise ImportError(
            "rdkit is not installed. To compute fingerprints, please install it via `pip install rdkit`."
        ) from None

    try:
        mmol = Chem.MolFromSmiles(smiles, sanitize=True)
    except:
        return None

    mfpgen = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(mfpgen.GetFingerprint(mmol))


def get_molecular_fingerprints(
    adata,
    compound_key: str,
    uns_key: str | None = None,
    smiles_key: str = "smiles",
    radius: int = 4,
    n_bits: int = 1024,
    copy: bool = False,
):
    """Computes Morgan fingerprints for compounds in `adata` and stores them in `adata.uns`.

    Args:
        adata: Annotated data matrix.
        compound_key: Key in `adata.obs` containing the compound identifiers.
        uns_key: Key in `adata.uns` to store the fingerprints.
        smiles_key: Key in `adata.obs` containing the SMILES representations of the compounds.
        radius: Radius of the Morgan fingerprint.
        n_bits: Number of bits in the fingerprint.

    Returns
    -------
        Updates `adata.uns` with the computed fingerprints.

        Sets the following fields:
        `.uns[uns_key]`: Dictionary containing the fingerprints for each compound.
    """
    adata = adata.copy() if copy else adata

    if uns_key is None:
        uns_key = f"{compound_key}_fingerprints"

    smiles_dict = adata.obs.set_index(compound_key)[smiles_key].to_dict()

    valid_fingerprints = {}
    not_found = []
    for comp, smiles in smiles_dict.items():
        comp_fp = _get_fingerprint(smiles, radius=radius, n_bits=n_bits)
        if comp_fp is not None:
            valid_fingerprints[comp] = comp_fp
        else:
            not_found.append(comp)

    if not_found:
        logger.warning(
            f"Could not compute fingerprints for the following compounds: {', '.join(not_found)}"
        )

    adata.uns[uns_key] = valid_fingerprints

    if copy:
        return adata


def encode_onehot(
    adata: ad.AnnData,
    covariate_keys: str | Sequence[str],
    uns_key: Sequence[str],
    exclude_values: str | Sequence[Any] = None,
    copy: bool = False,
) -> None | ad.AnnData:
    """Encodes covariates `adata.obs` as one-hot vectors and stores them in `adata.uns`.

    Args:
        adata: Annotated data matrix.
        covariate_keys: Keys of the covariates to encode.
        uns_key: Key in `adata.uns` to store the one-hot encodings.
        exclude_values: Values to exclude from encoding. These would usually be the control values.
        copy: Return a copy of `adata` instead of updating it in place.

    Returns
    -------
        If `copy` is `True`, returns a new `AnnData` object with the one-hot encodings stored in `adata.uns`. Otherwise, updates `adata` in place.

        Sets the following fields:
        `.uns[uns_key]`: Dictionary containing the one-hot encodings for each covariate.
    """
    adata = adata.copy() if copy else adata

    covariate_keys = _to_list(covariate_keys)
    exclude_values = _to_list(exclude_values)

    all_values = np.unique(adata.obs[covariate_keys].values.flatten())
    values_encode = np.setdiff1d(all_values, exclude_values).reshape(-1, 1)
    encoder = preprocessing.OneHotEncoder(sparse_output=False)
    encodings = encoder.fit_transform(values_encode)

    adata.uns[uns_key] = {}
    for value, encoding in zip(values_encode, encodings, strict=False):
        adata.uns[uns_key][value[0]] = encoding

    if copy:
        return adata
