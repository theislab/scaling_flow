from typing import Any


def _to_list(x: Any) -> list | tuple:
    """Converts x to a list if it is not already a list or tuple."""
    if isinstance(x, (list | tuple)):
        return x
    return [x]
