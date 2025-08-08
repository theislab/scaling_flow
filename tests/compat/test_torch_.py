import importlib

import pytest


class ImportBlocker:
    """Block importing of specific top-level packages.

    Inserts into sys.meta_path to raise ImportError for names starting with
    blocked_prefix.
    """

    def __init__(self, blocked_prefix: str):
        self.blocked_prefix = blocked_prefix

    def find_spec(self, fullname, path, target=None):
        if fullname == self.blocked_prefix or fullname.startswith(f"{self.blocked_prefix}."):
            raise ImportError(f"blocked import: {fullname}")
        return None


def test_torch_iterabledataset_fallback_raises(monkeypatch):
    # Block importing torch to trigger fallback path
    import sys

    blocker = ImportBlocker("torch")
    monkeypatch.setattr(sys, "meta_path", [blocker] + sys.meta_path, raising=False)

    # Ensure module is re-imported fresh
    if "cellflow.compat.torch_" in sys.modules:
        del sys.modules["cellflow.compat.torch_"]

    torch_mod = importlib.import_module("cellflow.compat.torch_")

    # Fallback class should be defined locally and raise on init
    from cellflow._optional import OptionalDependencyNotAvailable

    with pytest.raises(OptionalDependencyNotAvailable) as excinfo:
        _ = torch_mod.TorchIterableDataset()  # type: ignore[call-arg]
    assert "Optional dependency 'torch'" in str(excinfo.value)


def test_torch_iterabledataset_when_torch_available(monkeypatch):
    torch = pytest.importorskip("torch")

    # Make sure previously inserted blockers are not present by fully
    # reloading module
    mod_name = "cellflow.compat.torch_"
    if mod_name in list(importlib.sys.modules):
        del importlib.sys.modules[mod_name]
    compat_torch = importlib.import_module(mod_name)

    # Should alias to torch.utils.data.IterableDataset
    from torch.utils.data import IterableDataset as TorchIterableDatasetReal

    assert compat_torch.TorchIterableDataset is TorchIterableDatasetReal
