import matplotlib.pyplot as plt
import pytest

from cfp.plotting import plot_embeddings


class TestCallbacks:
    @pytest.mark.parametrize(
        "embedding", ["raw_embedding", "UMAP", "PCA", "Kernel_PCA"]
    )
    @pytest.mark.parametrize("dimensions", [(0, 1), (4, 5)])
    @pytest.mark.parametrize("hue", ["dose", None])
    def test_plot_embeddings(
        self, adata_with_condition_embedding, embedding, dimensions, hue
    ):
        fig = plot_embeddings(
            adata_with_condition_embedding,
            embedding=embedding,
            dimensions=dimensions,
            hue=hue,
        )

        assert isinstance(fig, plt.Figure)
