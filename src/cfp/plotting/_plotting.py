from typing import Any

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

from cfp import _constants
from cfp.model import CellFlow
from cfp.plotting._utils import _get_colors, _input_to_adata, get_plotting_vars


def plot_embeddings(
    obj: ad.AnnData | CellFlow,
    key: str = _constants.COVARIATE_EMBEDDING,
    labels: list[str] = None,
    col_dict: dict[str, str] | None = None,
    title: str | None = None,
    show_lines: bool = False,
    show_text: bool = False,
    return_fig: bool = True,
    **kwargs: Any,
) -> mpl.figure.Figure:

    circle_size = kwargs.pop("circle_size", 40)
    circe_transparency = kwargs.pop("circe_transparency", 1.0)
    line_transparency = kwargs.pop("line_transparency", 0.8)
    line_width = kwargs.pop("line_width", 1.0)
    fontsize = kwargs.pop("fontsize", 9)
    fig_width = kwargs.pop("fig_width", 4)
    fig_height = kwargs.pop("fig_height", 4)
    labels_name = kwargs.pop("labels_name", None)
    axis_equal = kwargs.pop("axis_equal", None)

    adata = _input_to_adata(obj)
    emb = get_plotting_vars(adata, _constants.COVARIATE_EMBEDDING, key=key)
    sns.set_style("white")

    # create data structure suitable for embedding
    df = pd.DataFrame(emb, columns=["dim1", "dim2"])
    if labels is not None:
        if labels_name is None:
            labels_name = "labels"
        df[labels_name] = labels

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()

    sns.despine(left=False, bottom=False, right=True)

    if (col_dict is None) and labels is not None:
        col_dict = _get_colors(labels)

    sns.scatterplot(
        x="dim1",
        y="dim2",
        hue=labels_name,
        palette=col_dict,
        alpha=circe_transparency,
        edgecolor="none",
        s=circle_size,
        data=df,
        ax=ax,
    )

    if show_lines:
        for i in range(len(emb)):
            if col_dict is None:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=None,
                )
            else:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=col_dict[labels[i]],
                )

    if show_text and labels is not None:
        texts = []
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx_label = np.where(labels == label)[0]
            texts.append(
                ax.text(
                    np.mean(emb[idx_label, 0]),
                    np.mean(emb[idx_label, 1]),
                    label,
                    fontsize=fontsize,
                )
            )

        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.1),  # noqa: C408
            ax=ax,
        )

    if axis_equal:
        ax.axis("equal")
        ax.axis("square")

    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold")

    ax.set_xlabel("dim1", fontsize=fontsize)
    ax.set_ylabel("dim2", fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    return fig if return_fig else None
