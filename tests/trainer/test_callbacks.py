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
        from cfp.training.callbacks import ComputePCADecodedMetrics, PCADecoder

        decoded_metrics_callback = ComputePCADecodedMetrics(
            metrics=metrics,
            pca_decoder=PCADecoder(pcs=adata_pca.varm['PCs'], means=adata_pca.varm["X_mean"])
        )

        reconstruction = decoded_metrics_callback.reconstruct_data(adata_pca.obsm["X_pca"])
        assert reconstruction.shape == adata_pca.X.shape
        assert jnp.allclose(reconstruction, adata_pca.layers["counts"])
