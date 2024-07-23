import abc
from typing import Any, Literal

import jax.tree_util as jtu
import numpy as np

from cfp._constants import ArrayLike
from cfp.data.data import ValidationData
from cfp.metrics.metrics import (
    compute_e_distance,
    compute_r_squared,
    compute_scalar_mmd,
    compute_sinkhorn_div,
)
from cfp.networks import ConditionalVelocityField


class BaseCallback(abc.ABC):

    @abc.abstractmethod
    def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abc.abstractmethod
    def on_log_iteration(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def on_train_end(self, *args: Any, **kwargs: Any) -> Any:
        pass


class LoggingCallback(BaseCallback, abc.ABC):
    @abc.abstractmethod
    def on_log_iteration(self, dict_to_log: dict[str, Any], **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def on_train_end(self, dict_to_log: dict[str, Any], **kwargs: Any) -> Any:
        pass


class ComputationCallback(BaseCallback, abc.ABC):
    @abc.abstractmethod
    def on_log_iteration(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
        **kwargs: Any,
    ) -> Any:
        pass

    def on_train_end(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
        **kwargs: Any,
    ) -> Any:
        pass


metric_to_func = {
    "r_squared": compute_r_squared,
    "MMD": compute_scalar_mmd,
    "sinkhorn_div": compute_sinkhorn_div,
    "e_distance": compute_e_distance,
}


class ComputeMetrics(ComputationCallback):
    def __init__(
        self,
        metrics: list[Literal["r_squared", "MMD", "sinkhorn_div", "e_distance"]],
        metric_aggregations: list[Literal["mean", "median", "id"]] = ["mean"],
    ):
        self.metrics = metrics
        self.metric_aggregations = metric_aggregations
        for metric in metrics:
            if metric not in metric_to_func:
                raise ValueError(
                    f"Metric {metric} not supported. Supported metrics are {list(metric_to_func.keys())}"
                )

    def on_train_begin(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def on_log_iteration(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
        **_: Any,
    ) -> dict[str, float]:
        metrics = {}
        for metric in self.metrics:
            for k in validation_data.keys():
                metrics[f"{k}_{metric}"] = jtu.tree_map(
                    metric_to_func[metric],
                    validation_data[k].tgt_data,
                    predicted_data[k],
                )
        # TODO: implement metric aggregation
        return metrics

    def on_train_end(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
        **_: Any,
    ) -> dict[str, float]:
        return self.on_log_iteration(validation_data, predicted_data)


class LoggingWandb(LoggingCallback):
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

    def on_log_iteration(
        self,
        dict_to_log: dict[str, float],
        **_: Any,
    ) -> Any:
        wandb.log(dict_to_log)

    def on_train_end(self, dict_to_log: dict[str, float]) -> Any:
        wandb.log(dict_to_log)


class CallbackRunner:
    def __init__(
        self,
        computation_callbacks: list[ComputationCallback],
        logging_callbacks: list[LoggingCallback],
        data: dict[str, ValidationData],
        n_conditions_train: int = 0,
        n_conditions_test: int = -1,
        n_conditions_ood: int = -1,
        seed: int = 0,
    ) -> None:
        for callback in computation_callbacks:
            if not isinstance(callback, ComputationCallback):
                raise ValueError(
                    f"Callback {callback} is not a subclass of `ComputationCallback`"
                )

        for callback in logging_callbacks:
            if not isinstance(callback, LoggingCallback):
                raise ValueError(
                    f"Callback {callback} is not a subclass of `LoggingCallback`"
                )

        self.computation_callbacks = self.computation_callbacks
        self.logging_callbacks = self.logging_callbacks
        self.validation_data = data
        self.n_conditions_train = n_conditions_train
        self.n_conditions_test = n_conditions_test
        self.n_conditions_ood = n_conditions_ood
        self.rng = np.random.default_rng(seed=seed)

    def _sample_conditions(
        self, dict_to_sample: dict[str, ArrayLike], n_conditions: int
    ) -> dict[str, ArrayLike]:
        if n_conditions == -1:
            return dict_to_sample
        else:
            keys = list(dict_to_sample.keys())
            idxs = self.rng.choice(len(keys), n_conditions, replace=False)
            new_d = {keys[i]: dict_to_sample[keys[i]] for i in idxs}
            return new_d

    def on_train_begin(self, *args: Any, **kwargs: Any) -> Any:
        for callback in self.computation_callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_log_iteration(
        self,
        cond_velocity_field: ConditionalVelocityField,
    ) -> dict[str, Any]:
        train_data_true = self._sample_conditions(
            train_data_true, self.n_conditions_train
        )
        test_data_true = self._sample_conditions(test_data_true, self.n_conditions_test)
        ood_data_true, ood_data_pred = self._sample_conditions(ood_data_true)

        train_data_pred = cond_velocity_field(train_data_true)  # TODO: adapt
        test_data_pred = cond_velocity_field(test_data_true)  # TODO: adapt
        ood_data_pred = cond_velocity_field(ood_data_true)  # TODO: adapt

        dict_to_log: dict[str, Any] = {}
        for callback in self.computation_callbacks:
            dict_to_log.update(
                callback.on_log_iteration(
                    cond_velocity_field=cond_velocity_field,
                    train_data_true=train_data_true,
                    test_data_true=test_data_true,
                    ood_data_true=ood_data_true,
                    train_data_pred=train_data_pred,
                    test_data_pred=test_data_pred,
                    ood_data_pred=ood_data_pred,
                )
            )

        for callback in self.logging_callbacks:
            callback.on_log_iteration(dict_to_log)

        return dict_to_log
