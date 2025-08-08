import sys
from dataclasses import dataclass

import numpy as np
import pytest

# Skip these tests entirely if torch is not available because the module
# under test imports torch at module import time.
pytest.importorskip("torch")

from cellflow.data._torch_dataloader import (  # noqa: E402
    CombinedTrainingSampler,
    _worker_init_fn_helper,
)


@dataclass
class DummySampler:
    label: str

    def sample(self, rng: np.random.Generator):  # noqa: D401
        return {"label": self.label, "rand": rng.random()}


def test_combined_sampler_requires_rng():
    s = CombinedTrainingSampler([DummySampler("a"), DummySampler("b")])
    with pytest.raises(ValueError):
        next(iter(s))


def test_combined_sampler_respects_weights_choice_first():
    s = CombinedTrainingSampler([DummySampler("a"), DummySampler("b")], weights=np.array([1.0, 0.0]))
    s.set_rng(np.random.default_rng(123))
    batch = next(iter(s))
    assert batch["label"] == "a"


def test_combined_sampler_respects_weights_choice_second():
    s = CombinedTrainingSampler([DummySampler("a"), DummySampler("b")], weights=np.array([0.0, 1.0]))
    s.set_rng(np.random.default_rng(123))
    batch = next(iter(s))
    assert batch["label"] == "b"


class _FakeDataset:
    def __init__(self):
        self._rng = None

    def set_rng(self, rng):
        self._rng = rng


def test_worker_init_fn_helper_sets_rng(monkeypatch):
    # Provide a fake torch with minimal API for get_worker_info
    class _FakeWorkerInfo:
        def __init__(self):
            self.id = 0
            self.dataset = _FakeDataset()

    class _FakeTorch:
        class utils:
            class data:
                @staticmethod
                def get_worker_info():
                    return _FakeWorkerInfo()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    rngs = [np.random.default_rng(42)]
    out = _worker_init_fn_helper(0, rngs)
    # Verify returned rng is the same and dataset received it
    assert out is rngs[0]
    assert _FakeTorch.utils.data.get_worker_info().dataset._rng is rngs[0]


