from collections.abc import Iterable, Mapping
from typing import Any

import anndata as ad
import zarr
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec


def write_sharded(
    group: zarr.Group,
    data: dict[str, Any],
    name: str,
    chunk_size: int = 4096,
    shard_size: int = 65536,
    compressors: Iterable[BytesBytesCodec] = (
        BloscCodec(
            cname="lz4",
            clevel=3,
        ),
    ),
):
    """Function to write data to a zarr group in a sharded format.

    Parameters
    ----------
    group
        The zarr group to write to.
    data
        The data to write.
    chunk_size
        The chunk size.
    shard_size
        The shard size.
    """
    # TODO: this is a copy of the function in arrayloaders
    # when it is no longer public we should use the function from arrayloaders
    # https://github.com/laminlabs/arrayloaders/blob/main/arrayloaders/io/store_creation.py
    ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr

    def get_size(shape: tuple[int, ...], chunk_size: int, shard_size: int) -> tuple[int, int]:
        shard_size_used = shard_size
        chunk_size_used = chunk_size
        if chunk_size > shape[0]:
            chunk_size_used = shard_size_used = shape[0]
        elif chunk_size < shape[0] or shard_size > shape[0]:
            chunk_size_used = shard_size_used = shape[0]
        return chunk_size_used, shard_size_used

    def callback(
        func: ad.experimental.Write,
        g: zarr.Group,
        k: str,
        elem: ad.typing.RWAble,
        dataset_kwargs: Mapping[str, Any],
        iospec: ad.experimental.IOSpec,
    ):
        if iospec.encoding_type in {"array"}:
            # Calculate greatest common divisor for first dimension
            # or use smallest dimension as chunk size

            chunk_size_used, shard_size_used = get_size(elem.shape, chunk_size, shard_size)
            
            dataset_kwargs = {
                "shards": (shard_size_used,) + (elem.shape[1:]),  # only shard over 1st dim
                "chunks": (chunk_size_used,) + (elem.shape[1:]),  # only chunk over 1st dim
                "compressors": compressors,
                **dataset_kwargs,
            }
        elif iospec.encoding_type in {"csr_matrix", "csc_matrix"}:
            dataset_kwargs = {
                "shards": (shard_size,),
                "chunks": (chunk_size,),
                "compressors": compressors,
                **dataset_kwargs,
            }

        func(g, k, elem, dataset_kwargs=dataset_kwargs)

    ad.experimental.write_dispatched(group, name, data, callback=callback)
    zarr.consolidate_metadata(group.store)


def _to_list(x: list[Any] | tuple[Any] | Any) -> list[Any] | tuple[Any]:
    """Converts x to a list if it is not already a list or tuple."""
    if isinstance(x, (list | tuple)):
        return x
    return [x]


def _flatten_list(x: Iterable[Iterable[Any]]) -> list[Any]:
    """Flattens a list of lists."""
    return [item for sublist in x for item in sublist]
