import numpy as np
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix

__all__ = ["centered_pca", "reconstruct_pca", "project_pca"]


def centered_pca(
    adata: ad.AnnData,
    n_comps: int = 50,
    layer: str | None = None,
    copy: bool = False,
    **kwargs,
):
    """Performs PCA on the centered data matrix and stores the results in `adata.obsm`.

    Parameters
    ----------
    adata : ad.AnnData
        An :class:`~anndata.AnnData` object.
    n_comps : int
        Number of principal components to compute.
    layer : str
        Layer in `adata.layers` to use for PCA.
    copy : bool
        Return a copy of `adata` instead of updating it in place.
    kwargs : dict
        Additional arguments to pass to `scanpy.pp.pca`.

    Returns
    -------
        If `copy` is `True`, returns a new `AnnData` object with the PCA results stored in `adata.obsm`. Otherwise, updates `adata` in place.

        Sets the following fields:
        `.obsm["X_pca"]`: PCA coordinates.
        `.varm["PCs"]`: Principal components.
        `.varm["X_mean"]`: Mean of the data matrix.
        `.layers["X_centered"]`: Centered data matrix.
        `.uns['pca']['variance_ratio']`: Variance ratio of each principal component.
        `.uns['pca']['variance']`: Variance of each principal component.
    """
    adata = adata.copy() if copy else adata
    X = adata.X if layer in [None, "X"] else adata.layers[layer]

    adata.varm["X_mean"] = X.mean(axis=0).T
    adata.layers["X_centered"] = csr_matrix(adata.X - adata.varm["X_mean"].T)
    sc.pp.pca(
        adata,
        n_comps=n_comps,
        layer="X_centered",
        zero_center=False,
        copy=False,
        **kwargs,
    )

    if copy:
        return adata


def reconstruct_pca(
    query_adata: ad.AnnData,
    use_rep: str = "X_pca",
    ref_adata: ad.AnnData | None = None,
    ref_means: np.ndarray | None = None,
    ref_pcs: np.ndarray | None = None,
    copy: bool = False,
):
    """Performs PCA on the data matrix and projects the data to the principal components.

    Parameters
    ----------
    query_adata : ad.AnnData
        An :class:`~anndata.AnnData` object with the query data.
    use_rep : str
        Representation to use for PCA. If `X`, uses `adata.X`. Otherwise, uses `adata.obsm[use_rep]`.
    ref_adata : ad.AnnData
        An :class:`~anndata.AnnData` object with the reference data.
    ref_means : np.ndarray
        Mean of the reference data. Only used if `ref_adata` is `None`.
    ref_pcs : np.ndarray
        Principal components of the reference data. Only used if `ref_adata` is `None`.
    copy : bool
        Return a copy of `adata` instead of updating it in place.

    Returns
    -------
        If `copy` is `True`, returns a new `AnnData` object with the PCA results stored in `adata.obsm`. Otherwise, updates `adata` in place.

        Sets the following fields:
        `.layers["X_recon"]`: Reconstructed data matrix.
    """
    if copy:
        query_adata = query_adata.copy()

    if (ref_adata is None) and ((ref_means is None) or (ref_pcs is None)):
        raise ValueError(
            "Either `ref_adata` or `ref_means` and `ref_pcs` must be provided."
        )

    X = query_adata.X if use_rep == "X" else query_adata.obsm[use_rep]
    if ref_adata is not None:
        ref_means = ref_adata.varm["X_mean"]
        ref_pcs = ref_adata.varm["PCs"]

    X_recon = np.array(X @ np.transpose(ref_pcs) + np.transpose(ref_means))
    query_adata.layers["X_recon"] = X_recon

    if copy:
        return query_adata


def project_pca(
    query_adata: ad.AnnData,
    ref_adata: ad.AnnData | None = None,
    ref_means: np.ndarray | None = None,
    ref_pcs: np.ndarray | None = None,
    layer: str | None = None,
    copy: bool = False,
):
    """Projects the query data to the principal components of the reference data.

    Parameters
    ----------
    query_adata : ad.AnnData
        An :class:`~anndata.AnnData` object with the query data.
    ref_adata : ad.AnnData
        An :class:`~anndata.AnnData` object with the reference data containing `adata.varm["X_mean"]` and `adata.varm["PCs"]`.
    ref_means : np.ndarray
        Mean of the reference data. Only used if `ref_adata` is `None`.
    ref_pcs : np.ndarray
        Principal components of the reference data. Only used if `ref_adata` is `None`.
    layer : str
        Layer in `adata.layers` to use for PCA.

    Returns
    -------
        If `copy` is `True`, returns a new `AnnData` object with the PCA results stored in `adata.obsm`. Otherwise, updates `adata` in place.

        Sets the following fields:
        `.obsm["X_pca"]`: PCA coordinates.
    """
    if copy:
        query_adata = query_adata.copy()

    if (ref_adata is None) and ((ref_means is None) or (ref_pcs is None)):
        raise ValueError(
            "Either `ref_adata` or `ref_means` and `ref_pcs` must be provided."
        )

    X = query_adata.X if layer in [None, "X"] else query_adata.layers[layer]
    if ref_adata is not None:
        ref_means = ref_adata.varm["X_mean"]
        ref_pcs = ref_adata.varm["PCs"]

    query_adata.obsm["X_pca"] = np.array((X - np.transpose(ref_means)) @ ref_pcs)

    if copy:
        return query_adata
