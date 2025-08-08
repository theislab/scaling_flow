from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from cellflow.compat import TorchIterableDataset
from cellflow.data._data import ZarrTrainingData
from cellflow.data._dataloader import TrainSampler


def _worker_init_fn_helper(worker_id, random_generators):
    import torch

    del worker_id
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id  # type: ignore[union-attr]
    rng = random_generators[worker_id]
    worker_info.dataset.set_rng(rng)  # type: ignore[union-attr]
    return rng


@dataclass
class TorchCombinedTrainSampler(TorchIterableDataset):
    """
    Combined training sampler that iterates over multiple samplers.

    Need to call set_rng(rng) before using the sampler.

    Args:
        samplers: List of training samplers.
        rng: Random number generator.
    """

    samplers: list[TrainSampler]
    weights: np.ndarray | None = None
    rng: np.random.Generator | None = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones(len(self.samplers))
        assert len(self.weights) == len(self.samplers)
        self.weights = self.weights / self.weights.sum()

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        if self.rng is None:
            raise ValueError("Please call set_rng() before using the sampler.")
        return self.samplers[self.rng.choice(len(self.samplers), p=self.weights)].sample(self.rng)

    @classmethod
    def combine_zarr_training_samplers(
        cls,
        data_paths: list[str],
        batch_size: int = 1024,
        seed: int = 42,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        weights: np.ndarray | None = None,
    ):
        seq = np.random.SeedSequence(seed)
        random_generators = [np.random.default_rng(s) for s in seq.spawn(len(data_paths))]
        worker_init_fn = partial(_worker_init_fn_helper, random_generators=random_generators)
        data = [ZarrTrainingData.read_zarr(path) for path in data_paths]
        samplers = [TrainSampler(data[i], batch_size) for i in range(len(data))]
        combined_sampler = cls(samplers, weights=weights)
        return torch.utils.data.DataLoader(
            combined_sampler,
            batch_size=None,
            worker_init_fn=worker_init_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
