import abc
from typing import Any, Literal

import jax
import jax.tree_util as jtu
import numpy as np

from cfp._constants import ArrayLike
from cfp.metrics.metrics import compute_e_distance, compute_r_squared, compute_scalar_mmd, compute_sinkhorn_div


class CFCallback(abc.ABC):

    @abc.abstractmethod
    def on_train_begin(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def on_log_iteration(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def on_train_end(self, *args: Any, **kwargs: Any) -> Any:
        pass


metric_to_func = {
    "r_squared": compute_r_squared,
    "MMD": compute_scalar_mmd,
    "sinkhorn_div": compute_sinkhorn_div,
    "e_distance": compute_e_distance,
}


class ComputeMetrics(CFCallback):
    def __init__(
        self,
        metrics: list[Literal["r_squared", "MMD", "sinkhorn_div", "e_distance"]],
        metric_aggregation: Literal["mean", "median", "id"] = "mean",
        n_conditions_train: int = -1,
        n_conditions_test: int = 1,
        n_conditions_ood: int = 1,
        seed: int = 0,
    ):
        self.metrics = metrics
        for metric in metrics:
            if metric not in metric_to_func:
                raise ValueError(
                    f"Metric {metric} not supported. Supported metrics are {list(metric_to_func.keys())}"
                )
        self.n_conditions_train = n_conditions_train
        self.n_conditions_test = n_conditions_test
        self.n_conditions_ood = n_conditions_ood
        self.rng = np.random.default_rng(seed=seed)

    def _sample_conditions(
        self, d1: dict[str, ArrayLike], d2: dict[str, ArrayLike], n_conditions: int
    ):
        if n_conditions == -1:
            return d1, d2
        else:
            keys = list(d1.keys())
            idxs = self.rng.choice(len(keys), n_conditions, replace=False)
            new_d1 = {keys[i]: d1[keys[i]] for i in idxs}
            new_d2 = {keys[i]: d2[keys[i]] for i in idxs}
            return new_d1, new_d2

    def on_train_begin(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def on_log_iteration(
        self,
        train_data_true: dict[str, Any],
        test_data_true: dict[str, Any],
        ood_data_true: dict[str, Any],
        train_data_pred: dict[str, jax.Array],
        test_data_pred: dict[str, Any],
        ood_data_pred: dict[str, Any],
    ) -> dict[str, float]:
        train_data_true, train_data_pred = self._sample_conditions(
            train_data_true, train_data_pred, self.n_conditions_train
        )
        test_data_true, test_data_pred = self._sample_conditions(
            test_data_true, test_data_pred, self.n_conditions_test
        )
        ood_data_true, ood_data_pred = self._sample_conditions(
            ood_data_true, ood_data_pred, self.n_conditions_ood
        )
        train_metrics = {}
        for metric in self.metrics:
            train_metrics[metric] = jtu.tree_map(
                metric_to_func[metric], train_data_true, train_data_pred
            )
        test_metrics = {}
        for metric in self.metrics:
            test_metrics[metric] = jtu.tree_map(
                metric_to_func[metric], test_data_true, test_data_pred
            )
        ood_metrics = {}
        for metric in self.metrics:
            ood_metrics[metric] = jtu.tree_map(
                metric_to_func[metric], ood_data_true, ood_data_pred
            )

    def on_train_end(self, *args: Any, **kwargs: Any) -> Any:
        self.compute_metrics()


class WandbLog(CFCallback):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb is not installed, please install it via `pip install wandb`"
        )
    try:
        import omegaconf
    except ImportError:
        raise ImportError(
            "omegaconf is not installed, please install it via `pip install omegaconf`"
        )

    def on_train_begin(
        self,
        wandb_project: str,
        out_dir: str,
        config_to_log: omegaconf.OmegaConf | dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        if isinstance(config_to_log, dict):
            config_to_log = omegaconf.OmegaConf.create(config_to_log)
        wandb.login()
        wandb.init(
            project=wandb_project,
            config=omegaconf.OmegaConf.to_container(config_to_log, resolve=True),
            dir=out_dir,
            settings=wandb.Settings(start_method=kwargs.pop("start_method", "thread")),
        )
