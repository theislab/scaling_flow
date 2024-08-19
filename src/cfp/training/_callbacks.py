import abc
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import jax.tree as jt
import jax.tree_util as jtu
import numpy as np
from numpy.typing import ArrayLike

from cfp.data._data import ValidationData
from cfp.metrics._metrics import compute_e_distance, compute_r_squared, compute_scalar_mmd, compute_sinkhorn_div

__all__ = [
    "BaseCallback",
    "LoggingCallback",
    "ComputationCallback",
    "Metrics",
    "WandbLogger",
    "CallbackRunner",
    "PCADecodedMetrics",
]


class BaseCallback(abc.ABC):
    """Base class for callbacks in the CellFlowTrainer"""

    @abc.abstractmethod
    def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
        """Called at the beginning of training"""
        pass

    @abc.abstractmethod
    def on_log_iteration(self, *args: Any, **kwargs: Any) -> Any:
        """Called at each validation/log iteration"""
        pass

    @abc.abstractmethod
    def on_train_end(self, *args: Any, **kwargs: Any) -> Any:
        """Called at the end of training"""
        pass


class LoggingCallback(BaseCallback, abc.ABC):
    """Base class for logging callbacks in the CellFlowTrainer"""

    @abc.abstractmethod
    def on_train_begin(self) -> Any:
        """Called at the beginning of training to initiate logging"""
        pass

    @abc.abstractmethod
    def on_log_iteration(self, dict_to_log: dict[str, Any]) -> Any:
        """Called at each validation/log iteration to log data

        Args:
            dict_to_log: Dictionary containing data to log
        """
        pass

    @abc.abstractmethod
    def on_train_end(self, dict_to_log: dict[str, Any]) -> Any:
        """Called at the end of trainging to log data

        Args:
            dict_to_log: Dictionary containing data to log
        """
        pass


class ComputationCallback(BaseCallback, abc.ABC):
    """Base class for computation callbacks in the CellFlowTrainer"""

    @abc.abstractmethod
    def on_train_begin(self) -> Any:
        """Called at the beginning of training to initiate metric computation"""
        pass

    @abc.abstractmethod
    def on_log_iteration(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        """Called at each validation/log iteration to compute metrics

        Args:
            validation_data: Validation data
            predicted_data: Predicted data
            training_data: Current batch and predicted data
        """
        pass

    @abc.abstractmethod
    def on_train_end(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        """Called at the end of training to compute metrics

        Args:
            validation_data: Validation data
            predicted_data: Predicted data
            training_data: Current batch and predicted data
        """
        pass


metric_to_func = {
    "r_squared": compute_r_squared,
    "mmd": compute_scalar_mmd,
    "sinkhorn_div": compute_sinkhorn_div,
    "e_distance": compute_e_distance,
}

agg_fn_to_func = {
    "mean": np.mean,
    "median": np.median,
}


class Metrics(ComputationCallback):
    """Callback to compute metrics on validation data during training

    Parameters
    ----------
    metrics : list
        List of metrics to compute
    metric_aggregation : list
        List of aggregation functions to use for each metric

    Returns
    -------
        None
    """

    def __init__(
        self,
        metrics: list[Literal["r_squared", "mmd", "sinkhorn_div", "e_distance"]],
        metric_aggregations: list[Literal["mean", "median"]] = None,
    ):
        self.metrics = metrics
        self.metric_aggregation = (
            ["mean"] if metric_aggregations is None else metric_aggregations
        )
        for metric in metrics:
            # TODO: support custom callables as metrics
            if metric not in metric_to_func:
                raise ValueError(
                    f"Metric {metric} not supported. Supported metrics are {list(metric_to_func.keys())}"
                )

    def on_train_begin(self, *args: Any, **kwargs: Any) -> Any:
        """Called at the beginning of training."""
        pass

    def on_log_iteration(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        """Called at each validation/log iteration to compute metrics

        Args:
            validation_data: Validation data
            predicted_data: Predicted data
        """
        metrics = {}
        for metric in self.metrics:
            for k in validation_data.keys():
                out = jtu.tree_map(
                    metric_to_func[metric], validation_data[k], predicted_data[k]
                )
                out_flattened = jt.flatten(out)[0]
                for agg_fn in self.metric_aggregation:
                    metrics[f"{k}_{metric}_{agg_fn}"] = agg_fn_to_func[agg_fn](
                        out_flattened
                    )

        return metrics

    def on_train_end(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        """Called at the end of training to compute metrics

        Args:
            validation_data: Validation data
            predicted_data: Predicted data
            training_data: Current batch and predicted data
        """
        return self.on_log_iteration(validation_data, predicted_data)


class PCADecoder(NamedTuple):
    pcs: ArrayLike
    means: ArrayLike


class PCADecodedMetrics(Metrics):
    """Callback to compute metrics on decoded validation data during training

    Parameters
    ----------
    pc_decoder
        Tuple of matrices containing the principal components and means to use for decoding
    metrics
        List of metrics to compute
    metric_aggregation
        List of aggregation functions to use for each metric
    means
        Means to use for decoding
    """

    def __init__(
        self,
        pca_decoder: PCADecoder,
        metrics: list[Literal["r_squared", "mmd", "sinkhorn_div", "e_distance"]],
        metric_aggregations: list[Literal["mean", "median"]] = None,
        log_prefix: str = "pca_decoded_",
    ):
        super().__init__(metrics, metric_aggregations)
        self.pcs = pca_decoder.pcs
        self.means = pca_decoder.means
        self.reconstruct_data = lambda x: x @ self.pcs.T + self.means.T
        self.log_prefix = log_prefix

    def on_log_iteration(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        """Called at each validation/log iteration to reconstruct the data and compute metrics on the reconstruction

        Args:
            validation_data: Validation data
            predicted_data: Predicted data
        """
        validation_data_decoded = jtu.tree_map(self.reconstruct_data, validation_data)
        predicted_data_decoded = jtu.tree_map(self.reconstruct_data, predicted_data)

        metrics = super().on_log_iteration(
            validation_data_decoded, predicted_data_decoded
        )
        metrics = {f"{self.log_prefix}{k}": v for k, v in metrics.items()}
        return metrics


class WandbLogger(LoggingCallback):
    """Callback to log data to Weights and Biases

    Parameters
    ----------
    project : str
        The project name in wandb
    out_dir : str
        The output directory to save the logs
    config : dict
        The configuration to log
    **kwargs : Any
        Additional keyword arguments to pass to wandb.init

    Returns
    -------
        None
    """

    def __init__(
        self,
        project: str,
        out_dir: str,
        config: dict[str, Any],
        **kwargs,
    ):
        self.project = project
        self.out_dir = out_dir
        self.config = config
        self.kwargs = kwargs

        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed, please install it via `pip install wandb`"
            ) from None
        try:
            import omegaconf

            self.omegaconf = omegaconf
        except ImportError:
            raise ImportError(
                "omegaconf is not installed, please install it via `pip install omegaconf`"
            ) from None

    def on_train_begin(self) -> Any:
        """Called at the beginning of training to initiate WandB logging"""
        if isinstance(self.config, dict):
            config = self.omegaconf.OmegaConf.create(self.config)
        self.wandb.login()
        self.wandb.init(
            project=self.project,
            config=self.omegaconf.OmegaConf.to_container(config, resolve=True),
            dir=self.out_dir,
            settings=self.wandb.Settings(
                start_method=self.kwargs.pop("start_method", "thread")
            ),
        )

    def on_log_iteration(
        self,
        dict_to_log: dict[str, float],
        **_: Any,
    ) -> Any:
        """Called at each validation/log iteration to log data to WandB"""
        self.wandb.log(dict_to_log)

    def on_train_end(self, dict_to_log: dict[str, float]) -> Any:
        """Called at the end of training to log data to WandB"""
        self.wandb.log(dict_to_log)


class CallbackRunner:
    """Runs a set of computational and logging callbacks in the CellFlowTrainer

    Args:
        computation_callbacks: List of computation callbacks
        logging_callbacks: List of logging callbacks
        seed: Random seed for subsampling the validation data

    Returns
    -------
        None
    """

    def __init__(
        self,
        callbacks: Sequence[BaseCallback],
    ) -> None:

        self.computation_callbacks: list[ComputationCallback] = [
            c for c in callbacks if isinstance(c, ComputationCallback)
        ]
        self.logging_callbacks: list[LoggingCallback] = [
            c for c in callbacks if isinstance(c, LoggingCallback)
        ]

        if len(self.computation_callbacks) == 0 & len(self.logging_callbacks) != 0:
            raise ValueError(
                "No computation callbacks defined to compute metrics to log"
            )

    def on_train_begin(self) -> Any:
        """Called at the beginning of training to initiate callbacks"""
        for callback in self.computation_callbacks:
            callback.on_train_begin()

        for callback in self.logging_callbacks:
            callback.on_train_begin()

    def on_log_iteration(
        self, valid_data: dict[str, ValidationData], pred_data
    ) -> dict[str, Any]:
        """Called at each validation/log iteration to run callbacks. First computes metrics with computation callbacks and then logs data with logging callbacks.

        Parameters
        ----------
        valid_data: Validation data
        pred_data: Predicted data corresponding to validation data

        Returns
        -------
            dict_to_log: Dictionary containing data to log
        """
        dict_to_log: dict[str, Any] = {}

        for callback in self.computation_callbacks:
            results = callback.on_log_iteration(valid_data, pred_data)
            dict_to_log.update(results)

        for callback in self.logging_callbacks:
            callback.on_log_iteration(dict_to_log)

        return dict_to_log

    def on_train_end(self, valid_data, pred_data) -> dict[str, Any]:
        """Called at the end of training to run callbacks. First computes metrics with computation callbacks and then logs data with logging callbacks.

        Args:
            valid_data: Validation data
            pred_data: Predicted data

        Returns
        -------
            dict_to_log: Dictionary containing data to log
        """
        dict_to_log: dict[str, Any] = {}

        for callback in self.computation_callbacks:
            results = callback.on_log_iteration(valid_data, pred_data)
            dict_to_log.update(results)

        for callback in self.logging_callbacks:
            callback.on_log_iteration(dict_to_log)

        return dict_to_log
