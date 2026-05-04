"""Common utilities for working with Zarr groups in consistent ways."""

import json
import warnings
from pathlib import Path
from typing import Literal, TypeAlias

import dask.array as da
import fsspec
import zarr
from pydantic_zarr.v2 import ArraySpec as AnyArraySpecV2
from pydantic_zarr.v3 import ArraySpec as AnyArraySpecV3
from zarr.abc.store import Store
from zarr.errors import ContainsGroupError
from zarr.storage import FsspecStore, LocalStore, MemoryStore

from ngio_tables._utils._errors import (
    NgioTablesFileExistsError,
    NgioTablesFileNotFoundError,
    NgioTablesValueError,
)
from ngio_tables._utils._warnings import NgioTablesUserWarning

AccessModeLiteral = Literal["r", "r+", "w", "w-", "a"]
# StoreLike is more restrictive than it could be
# but to make sure we can handle the store correctly
# we need to be more restrictive
NgioSupportedStore: TypeAlias = (
    str | Path | fsspec.mapping.FSMap | FsspecStore | MemoryStore | dict | LocalStore
)
GenericStore: TypeAlias = NgioSupportedStore | Store
StoreOrGroup: TypeAlias = NgioSupportedStore | zarr.Group


def _check_store(store) -> NgioSupportedStore:
    """Check the store and return a valid store."""
    if not isinstance(store, NgioSupportedStore):
        warnings.warn(
            f"Store type {type(store)} is not explicitly supported. "
            f"Supported types are: {NgioSupportedStore}. "
            "Proceeding, but this may lead to unexpected behavior.",
            NgioTablesUserWarning,
            stacklevel=2,
        )
    return store


def _check_group(
    group: zarr.Group, mode: AccessModeLiteral | None = None
) -> zarr.Group:
    """Check the group and return a valid group."""
    if group.read_only and mode not in [None, "r"]:
        raise NgioTablesValueError(
            f"The group is read only. Cannot open in mode {mode}."
        )

    if mode == "r" and not group.read_only:
        # let's make sure we don't accidentally write to the group
        group = zarr.open_group(store=group.store, path=group.path, mode="r")
    return group


def open_group_wrapper(
    store: StoreOrGroup,
    mode: AccessModeLiteral | None = None,
    zarr_format: Literal[2, 3] | None = None,
) -> zarr.Group:
    """Wrapper around zarr.open_group with some additional checks.

    Args:
        store (StoreOrGroup): The store or group to open.
        mode (AccessModeLiteral): The mode to open the group in.
        zarr_format (int): The Zarr format version to use.

    Returns:
        zarr.Group: The opened Zarr group.
    """
    if isinstance(store, zarr.Group):
        group = _check_group(store, mode)
        _check_store(group.store)
        return group

    try:
        _check_store(store)
        mode = mode if mode is not None else "a"
        group = zarr.open_group(store=store, mode=mode, zarr_format=zarr_format)

    except FileExistsError as e:
        raise NgioTablesFileExistsError(
            f"A Zarr group already exists at {store}, consider setting overwrite=True."
        ) from e

    except FileNotFoundError as e:
        raise NgioTablesFileNotFoundError(f"No Zarr group found at {store}") from e

    except ContainsGroupError as e:
        raise NgioTablesFileExistsError(
            f"A Zarr group already exists at {store}, consider setting overwrite=True."
        ) from e

    return group


def find_dimension_separator(array: zarr.Array) -> Literal[".", "/"]:
    """Find the dimension separator used in the Zarr store.

    Args:
        array (zarr.Array): The Zarr array to check.

    Returns:
        Literal[".", "/"]: The dimension separator used in the store.
    """
    from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding

    if array.metadata.zarr_format == 2:
        separator = array.metadata.dimension_separator  # ty:ignore[unresolved-attribute]
    else:
        separator = array.metadata.chunk_key_encoding  # ty:ignore[unresolved-attribute]
        if not isinstance(separator, DefaultChunkKeyEncoding):
            raise NgioTablesValueError(
                "Only DefaultChunkKeyEncoding is supported in this example."
            )
        separator = separator.separator
    return separator


def is_group_listable(group: zarr.Group) -> bool:
    """Check if a Zarr group is listable.

    A group is considered listable if it contains at least one array or subgroup.

    Args:
        group (zarr.Group): The Zarr group to check.

    Returns:
        bool: True if the group is listable, False otherwise.
    """
    if not group.store.supports_listing:
        # If the store does not support listing
        # then for sure it is not listable
        return False
    try:
        next(group.keys())
        return True
    except StopIteration:
        # Group is listable but empty
        return True
    except Exception as _:
        # Some stores may raise errors when listing
        # consider those not listable
        return False


def _make_sync_fs(fs: fsspec.AbstractFileSystem) -> fsspec.AbstractFileSystem:
    fs_dict = json.loads(fs.to_json())
    fs_dict["asynchronous"] = False
    return fsspec.AbstractFileSystem.from_json(json.dumps(fs_dict))


def _get_mapper(store: LocalStore | FsspecStore, path: str):
    if isinstance(store, LocalStore):
        fs = fsspec.filesystem("file")
        full_path = (store.root / path).as_posix()
    else:
        fs = _make_sync_fs(store.fs)
        full_path = f"{store.path}/{path}"
    return fs.get_mapper(full_path)


def _fsspec_copy(
    src_fs: LocalStore | FsspecStore,
    src_path: str,
    dest_fs: LocalStore | FsspecStore,
    dest_path: str,
):
    src_mapper = _get_mapper(src_fs, src_path)
    dest_mapper = _get_mapper(dest_fs, dest_path)
    for key in src_mapper.keys():
        dest_mapper[key] = src_mapper[key]


def _zarr_python_copy(src_group: zarr.Group, dest_group: zarr.Group):
    # Copy attributes
    dest_group.attrs.put(src_group.attrs.asdict())
    # Copy arrays
    for name, array in src_group.arrays():
        if array.metadata.zarr_format == 2:
            spec = AnyArraySpecV2.from_zarr(array)
        else:
            spec = AnyArraySpecV3.from_zarr(array)
        dst = spec.to_zarr(
            store=dest_group.store,
            path=f"{dest_group.path}/{name}",
            overwrite=True,
        )
        if array.ndim > 0:
            dask_array = da.from_zarr(array)
            da.to_zarr(dask_array, dst, overwrite=False)
    # Copy subgroups
    for name, subgroup in src_group.groups():
        dest_subgroup = dest_group.create_group(name, overwrite=True)
        _zarr_python_copy(subgroup, dest_subgroup)


def copy_group(
    src_group: zarr.Group, dest_group: zarr.Group, suppress_warnings: bool = False
):
    """Copy a Zarr group to another group.

    Args:
        src_group (zarr.Group): The source group.
        dest_group (zarr.Group): The destination group.
        suppress_warnings (bool): If True, suppress warnings about fallback copy.
    """
    if src_group.metadata.zarr_format != dest_group.metadata.zarr_format:
        raise NgioTablesValueError(
            "Different Zarr format versions between source and destination, "
            "cannot copy."
        )

    if not is_group_listable(src_group):
        raise NgioTablesValueError("Source group is not listable, cannot copy.")

    if dest_group.read_only:
        raise NgioTablesValueError("Destination group is read only, cannot copy.")
    if isinstance(src_group.store, LocalStore | FsspecStore) and isinstance(
        dest_group.store, LocalStore | FsspecStore
    ):
        _fsspec_copy(src_group.store, src_group.path, dest_group.store, dest_group.path)
        return
    if not suppress_warnings:
        warnings.warn(
            "Fsspec copy not possible, falling back to Zarr Python API for the copy. "
            "This will preserve some tabular data non-zarr native (parquet, and csv), "
            "and it will be slower for large datasets.",
            NgioTablesUserWarning,
            stacklevel=2,
        )
    _zarr_python_copy(src_group, dest_group)
