import sys
import threading

import numpy as np
import pytest

from cellflow.data._dataloader import TrainSampler
from cellflow.data._jax_dataloader import JaxOutOfCoreTrainSampler


class _DummyData:
    def __init__(self):
        self.cell_data = np.arange(20).reshape(10, 2)
        self.split_covariates_mask = np.array([0] * 5 + [1] * 5)
        self.perturbation_covariates_mask = np.array([0] * 5 + [1] * 5)
        self.control_to_perturbation = {0: np.array([0]), 1: np.array([1])}
        self.condition_data = None


def test_jax_out_of_core_sampler_no_jax(monkeypatch):
    # Skip if jax is installed; this test ensures no import errors when jax missing
    if "jax" in sys.modules:
        pytest.skip("JAX present in environment; skip missing-JAX behavior test")

    sampler = JaxOutOfCoreTrainSampler(data=_DummyData(), seed=0, batch_size=2, num_workers=1, prefetch_factor=1)
    # set_sampler imports jax; confirm it raises ImportError when jax not present
    with pytest.raises(ImportError):
        sampler.set_sampler(num_iterations=1)


@pytest.mark.skipif("jax" not in sys.modules, reason="Requires JAX runtime in environment")
def test_jax_out_of_core_sampler_with_jax(monkeypatch):
    # Basic smoke test when JAX is available
    data = _DummyData()
    sampler = JaxOutOfCoreTrainSampler(data=data, seed=0, batch_size=2, num_workers=1, prefetch_factor=1)
    sampler.set_sampler(num_iterations=2)
    b1 = sampler.sample()
    b2 = sampler.sample()
    assert set(b1.keys()) == {"src_cell_data", "tgt_cell_data"}
    assert b1["src_cell_data"].shape[0] == 2
    assert b2["src_cell_data"].shape[0] == 2


