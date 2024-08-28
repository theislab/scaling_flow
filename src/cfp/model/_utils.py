import anndata as ad
import jax
import jax.numpy as jnp

from cfp import _constants
from cfp._types import ArrayLike
from cfp.data.dataloader import PredictionData


def _multivariate_normal(
    rng: jax.Array,
    shape: tuple[int, ...],
    dim: int,
    mean: float = 0.0,
    cov: float = 1.0,
) -> jnp.ndarray:
    mean = jnp.full(dim, fill_value=mean)
    cov = jnp.diag(jnp.full(dim, fill_value=cov))
    return jax.random.multivariate_normal(rng, mean=mean, cov=cov, shape=shape)


def _write_predictions(
    adata: ad.AnnData,
    predictions: dict[str, dict[str, ArrayLike]] | dict[str, ArrayLike],
    pred_data: PredictionData,
    prefix_to_store: str = _constants.PREDICTION_PREFIX,
) -> None:

    pass
