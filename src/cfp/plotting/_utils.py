from collections.abc import Sequence
from typing import Any

import anndata as ad
import seaborn as sns

from cfp import _constants, _logging
from cfp.model import CellFlow


def _get_palette(
    n_colors: int, palette_name: str | None = "Set1"
) -> sns.palettes._ColorPalette:
    try:
        palette = sns.color_palette(palette_name)
    except ValueError:
        _logging.logger.info("Palette not found. Using default palette tab10")
        palette = sns.color_palette()
    while len(palette) < n_colors:
        palette += palette

    return palette


def _get_colors(
    labels: Sequence[str],
    palette: str | None = None,
    palette_name: str | None = None,
) -> dict[str, str]:
    n_colors = len(labels)
    if palette is None:
        palette = _get_palette(n_colors, palette_name)
    col_dict = dict(zip(labels, palette[:n_colors], strict=False))
    return col_dict


def get_plotting_vars(adata: ad.AnnData, func_key: str, *, key: str) -> Any:
    uns_key = _constants.CFP_KEY
    try:
        return adata.uns[uns_key][func_key][key]
    except KeyError:
        raise KeyError(
            f"No data found in `adata.uns[{uns_key!r}][{func_key!r}][{key!r}]`."
        ) from None


def _input_to_adata(obj: ad.AnnData | CellFlow) -> ad.AnnData:
    if isinstance(obj, ad.AnnData):
        return obj
    elif isinstance(obj, CellFlow):
        return obj.adata
    else:
        raise ValueError(
            f"obj must be an AnnData or CellFlow object, but found {type(obj)}"
        )
