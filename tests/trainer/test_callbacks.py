import anndata as ad
import jax.numpy as jnp
import pytest

class TestCallbacks:
    @pytest.mark.parametrize("metrics", [['r_squared']])
    def test_pca_reconstruction(
        self,
        adata_pca: ad.AnnData,
        metrics
    ):
        from cfp.training.callbacks import ComputePCADecodedMetrics

        decoded_metrics_callback = ComputePCADecodedMetrics(
            metrics=metrics,
            PCs=adata_pca.varm['PCs'], 
            means=adata_pca.varm["X_mean"])

        reconstruction = decoded_metrics_callback.reconstruct_data(adata_pca.obsm["X_pca"])
        assert reconstruction.shape == adata_pca.X.shape
        print(reconstruction)
        print(adata_pca.layers["counts"])
        print(adata_pca.varm["X_mean"])
        assert jnp.allclose(reconstruction, adata_pca.layers["counts"])
