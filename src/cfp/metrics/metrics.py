import numpy as np
from ott.geometry import costs, pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.metrics.pairwise import rbf_kernel

__all__ = ["compute_metrics", "compute_metrics_fast"]


def compute_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the R squared between true (x) and predicted (y)"""
    return r2_score(np.mean(x, axis=0), np.mean(y, axis=0))


def compute_sinkhorn_div(x: np.ndarray, y: np.ndarray, epsilon: float) -> float:
    """Compute the Sinkhorn divergence between x and y."""
    return float(
        sinkhorn_divergence(
            pointcloud.PointCloud,
            x=x,
            y=y,
            cost_fn=costs.SqEuclidean(),
            epsilon=epsilon,
            scale_cost=1.0,
        ).divergence
    )


def compute_e_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the energy distance as in Peidli et al."""
    sigma_X = pairwise_distances(x, x, metric="sqeuclidean").mean()
    sigma_Y = pairwise_distances(y, y, metric="sqeuclidean").mean()
    delta = pairwise_distances(x, y, metric="sqeuclidean").mean()
    return 2 * delta - sigma_X - sigma_Y


def compute_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute different metrics for x (true) and y (predicted)."""
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["sinkhorn_div_1"] = compute_sinkhorn_div(x, y, epsilon=1.0)
    metrics["sinkhorn_div_10"] = compute_sinkhorn_div(x, y, epsilon=10.0)
    metrics["sinkhorn_div_100"] = compute_sinkhorn_div(x, y, epsilon=100.0)
    metrics["e_distance"] = compute_e_distance(x, y)
    metrics["mmd"] = compute_scalar_mmd(x, y)
    return metrics


def compute_mean_metrics(metrics: dict[str, dict[str, float]], prefix: str = ""):
    """Compute the mean value of different metrics."""
    metric_names = list(list(metrics.values())[0].keys())
    metric_dict = {prefix + met_name: [] for met_name in metric_names}
    for met in metric_names:
        stat = 0.0
        for vals in metrics.values():
            stat += vals[met]
        metric_dict[prefix + met] = stat / len(metrics)
    return metric_dict


def mmd_distance(x, y, gamma):
    """Compute single MMD based on RBF kernel."""
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_scalar_mmd(target, transport, gammas=None):  # from CellOT repo
    """Compute MMD across different length scales"""
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(lambda x: safe_mmd(target, transport, x), gammas))


def compute_metrics_fast(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute metrics which are fast to compute."""
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["e_distance"] = compute_e_distance(x, y)
    metrics["mmd_distance"] = compute_scalar_mmd(x, y)
    return metrics
