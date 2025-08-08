import importlib

import pytest


def test_optional_dependency_exception_message():
    opt = importlib.import_module("cellflow._optional")
    # Ensure exception type exists and message contains installation hint
    with pytest.raises(opt.OptionalDependencyNotAvailable) as excinfo:
        raise opt.OptionalDependencyNotAvailable(opt.torch_required_msg())
    msg = str(excinfo.value)
    assert "Optional dependency 'torch' is required" in msg
    assert "pip install torch" in msg


