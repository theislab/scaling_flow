import anndata as ad
import jax.numpy as jnp
import pytest


class TestCallbacks:
    @pytest.mark.parametrize("metrics", [["r_squared"]])
    def test_pca_reconstruction(self, adata_pca: ad.AnnData, metrics):
        from cfp.training import PCADecodedMetrics

        decoded_metrics_callback = PCADecodedMetrics(
            metrics=metrics,
            ref_adata=adata_pca,
        )

        reconstruction = decoded_metrics_callback.reconstruct_data(
            adata_pca.obsm["X_pca"]
        )
        assert reconstruction.shape == adata_pca.X.shape
        assert jnp.allclose(reconstruction, adata_pca.layers["counts"])

    @pytest.mark.parametrize("metrics", [["r_squared"]])
    def test_vae_reconstruction(self, metrics):
        from scvi.data import synthetic_iid

        from cfp.external import CFJaxSCVI
        from cfp.training import VAEDecodedMetrics

        adata = synthetic_iid()
        CFJaxSCVI.setup_anndata(
            adata,
            batch_key="batch",
        )
        model = CFJaxSCVI(adata, n_latent=2, gene_likelihood="normal")
        model.train(2, train_size=0.5, check_val_every_n_epoch=1)
        out = model.get_latent_representation(give_mean=True)

        vae_decoded_metrics_callback = VAEDecodedMetrics(
            vae=model,
            metrics=metrics,
        )

        reconstruction = vae_decoded_metrics_callback.reconstruct_data(out)

        assert reconstruction.shape == adata.X.shape
