from typing import Any


def _to_list(x: Any) -> list | tuple:
    """Converts x to a list if it is not already a list or tuple."""
    if isinstance(x, (list | tuple)):
        return x
    return [x]


def _flatten_list(x: list | tuple) -> list:
    """Flattens a list of lists."""
    return [item for sublist in x for item in sublist]
