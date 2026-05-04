"""Module for handling the /tables group in an OME-NGFF file."""

from typing import Any, Literal, Protocol, TypeVar, cast

import anndata as ad
import pandas as pd
import polars as pl
import zarr

from ngio_tables._utils import (
    AccessModeLiteral,
    NgioTablesValidationError,
    NgioTablesValueError,
    StoreOrGroup,
    open_group_wrapper,
)
from ngio_tables.backends import (
    BackendMeta,
    DefaultTableBackend,
    TableBackend,
    TabularData,
)
from ngio_tables.v1 import (
    ConditionTableV1,
    FeatureTableV1,
    GenericTable,
    MaskingRoiTableV1,
    RoiTableV1,
)
from ngio_tables.v1._roi_table import _RoiTableBase

GenericRoiTable = _RoiTableBase
RoiTable = RoiTableV1
MaskingRoiTable = MaskingRoiTableV1
FeatureTable = FeatureTableV1
ConditionTable = ConditionTableV1


class Table(Protocol):
    """Placeholder class for a table."""

    @staticmethod
    def table_type() -> str:
        """Return the type of the table."""
        ...

    @staticmethod
    def version() -> str:
        """Return the version of the table."""
        ...

    @property
    def backend_name(self) -> str | None:
        """The name of the backend."""
        ...

    @property
    def meta(self) -> BackendMeta:
        """Return the metadata for the table."""
        ...

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the table as a DataFrame."""
        ...

    @property
    def lazy_frame(self) -> pl.LazyFrame:
        """Return the table as a LazyFrame."""
        ...

    @property
    def anndata(self) -> ad.AnnData:
        """Return the table as an AnnData object."""
        ...

    def set_table_data(
        self,
        table_data: TabularData | None = None,
        refresh: bool = False,
    ) -> None:
        """Make sure that the table data is set (exist in memory).

        If an object is passed, it will be used as the table.
        If None is passed, the table will be loaded from the backend.

        If refresh is True, the table will be reloaded from the backend.
            If table is not None, this will be ignored.
        """
        ...

    def set_backend(
        self,
        group: zarr.Group | None = None,
        backend: TableBackend = DefaultTableBackend,
    ) -> None:
        """Set the backend store and path for the table.

        Either a group or a backend must be provided.

        If group is None it will be inferred from the existing backend.
        If the backend is None, it will be inferred from the group attrs.
        """
        ...

    @classmethod
    def from_group(
        cls,
        group: zarr.Group,
        backend: TableBackend | None = None,
    ) -> "Table":
        """Create a new table from a zarr group."""
        ...

    @classmethod
    def from_table_data(cls, table_data: TabularData, meta: BackendMeta) -> "Table":
        """Create a new table from a DataFrame."""
        ...

    @property
    def table_data(self) -> TabularData:
        """Return the table."""
        ...

    def consolidate(self) -> None:
        """Consolidate the table on disk."""
        ...


TypedTable = Literal[
    "generic_table",
    "roi_table",
    "masking_roi_table",
    "feature_table",
    "condition_table",
]

TypedRoiTable = Literal[
    "roi_table",
    "masking_roi_table",
]

TableType = TypeVar("TableType", bound=Table)


class TableMeta(BackendMeta):
    """Base class for table metadata."""

    table_version: str = "1"
    type: str = "generic_table"

    def unique_name(self) -> str:
        """Return the unique name for the table."""
        return f"{self.type}_v{self.table_version}"


def _get_meta(group: zarr.Group) -> TableMeta:
    """Get the metadata from a zarr group."""
    attrs: dict[str, Any] = group.attrs.asdict()
    meta = TableMeta(**attrs)
    return meta


class ImplementedTables:
    """A singleton class to manage the available table handler plugins."""

    _instance = None
    _implemented_tables: dict[str, type[Table]]

    def __new__(cls):
        """Create a new instance of the class if it does not exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._implemented_tables = {}
        return cls._instance

    def available_implementations(self) -> list[str]:
        """Get the available table handler versions."""
        return list(self._implemented_tables.keys())

    def get_table(
        self,
        meta: TableMeta,
        group: zarr.Group,
        backend: TableBackend | None = None,
        strict: bool = True,
    ) -> Table:
        """Try to get a handler for the given store based on the metadata version."""
        if strict:
            default = None
        else:
            default = GenericTable

        table_cls = self._implemented_tables.get(meta.unique_name(), default)
        if table_cls is None:
            raise NgioTablesValueError(
                f"Table handler for {meta.unique_name()} not implemented."
            )
        table = table_cls.from_group(group=group, backend=backend)
        return table

    def _add_implementation(
        self, handler: type[Table], name: str, overwrite: bool = False
    ) -> None:
        """Register a new table handler."""
        if name in self._implemented_tables and not overwrite:
            raise NgioTablesValueError(
                f"Table handler for {name} already implemented. "
                "Use overwrite=True to replace it."
            )
        self._implemented_tables[name] = handler

    def add_implementation(
        self,
        handler: type[Table],
        overwrite: bool = False,
        aliases: list[str] | None = None,
    ) -> None:
        """Register a new table handler."""
        meta = TableMeta(
            type=handler.table_type(),
            table_version=handler.version(),
        )

        self._add_implementation(handler, meta.unique_name(), overwrite)

        if aliases is not None:
            for alias in aliases:
                self._add_implementation(handler, alias, overwrite)


class TablesContainer:
    """A class to handle the /tables group in an OME-NGFF file."""

    def __init__(self, group: zarr.Group) -> None:
        """Initialize the TablesContainer."""
        self._group = group

        # Validate the group
        # Either contains a tables attribute or is empty
        attrs = self._group.attrs.asdict()
        if len(attrs) == 0:
            # It's an empty group
            pass
        elif "tables" in attrs and isinstance(attrs["tables"], list):
            # It's a valid group
            pass
        else:
            raise NgioTablesValidationError(
                f"Invalid /tables group. "
                f"Expected a single tables attribute with a list of table names. "
                f"Found: {attrs}"
            )

    def _get_tables_list(self) -> list[str]:
        """Return the list of table names from the group attributes."""
        attrs: dict[str, Any] = self._group.attrs.asdict()
        return attrs.get("tables", [])

    def _get_table_group(self, name: str) -> zarr.Group:
        """Get the zarr group for a table."""
        return zarr.open_group(
            store=self._group.store,
            path=f"{self._group.path}/{name}",
            mode="r+",
        )

    def list(self, filter_types: TypedTable | str | None = None) -> list[str]:
        """List all tables in the group.

        Args:
            filter_types: If provided, only return tables of this type.

        Returns:
            A list of table names.
        """
        tables = self._get_tables_list()
        if filter_types is None:
            return tables

        filtered_tables = []
        for table_name in tables:
            table_group = self._get_table_group(table_name)
            table_type = _get_meta(table_group).type
            if table_type == filter_types:
                filtered_tables.append(table_name)
        return filtered_tables

    def get(
        self,
        name: str,
        backend: TableBackend | None = None,
        strict: bool = True,
    ) -> Table:
        """Get a table from the group.

        Args:
            name: The name of the table.
            backend: The backend to use for reading the table.
            strict: If True, raise an error if the table type is not implemented.

        Returns:
            The table object.
        """
        if name not in self.list():
            raise NgioTablesValueError(f"Table '{name}' not found in the group.")

        table_group = self._get_table_group(name)

        meta = _get_meta(table_group)
        return ImplementedTables().get_table(
            meta=meta,
            group=table_group,
            backend=backend,
            strict=strict,
        )

    def get_as(
        self,
        name: str,
        table_cls: type[TableType],
        backend: TableBackend | None = None,
    ) -> TableType:
        """Get a table from the group as a specific type.

        Args:
            name: The name of the table.
            table_cls: The table class to use for loading the table.
            backend: The backend to use for reading the table.

        Returns:
            The table object of the specified type.
        """
        if name not in self.list():
            raise NgioTablesValueError(f"Table '{name}' not found in the group.")

        table_group = self._get_table_group(name)
        return cast(
            "TableType",
            table_cls.from_group(
                group=table_group,
                backend=backend,
            ),
        )

    def delete(self, name: str, missing_ok: bool = False) -> None:
        """Delete a table from the group.

        Args:
            name (str): The name of the table to delete.
            missing_ok (bool): If True, do not raise an error if
                the table does not exist.
        """
        existing_tables = self._get_tables_list()
        if name not in existing_tables:
            if missing_ok:
                return
            raise NgioTablesValueError(
                f"Table '{name}' not found in the Tables group. "
                f"Available tables: {existing_tables}"
            )

        del self._group[name]
        existing_tables.remove(name)
        self._group.attrs.update({"tables": existing_tables})

    def add(
        self,
        name: str,
        table: Table,
        backend: TableBackend = DefaultTableBackend,
        overwrite: bool = False,
    ) -> None:
        """Add a table to the group.

        Args:
            name: The name of the table.
            table: The table object to add.
            backend: The backend to use for writing the table.
            overwrite: Whether to overwrite an existing table with the same name.
        """
        existing_tables = self._get_tables_list()
        if name in existing_tables and not overwrite:
            raise NgioTablesValueError(
                f"Table '{name}' already exists in the group. "
                "Use overwrite=True to replace it."
            )

        table_group = zarr.open_group(
            store=self._group.store,
            path=f"{self._group.path}/{name}",
            mode="w" if overwrite else "a",
        )

        if backend is None:
            backend = table.backend_name

        table.set_table_data()
        table.set_backend(
            group=table_group,
            backend=backend,
        )
        table.consolidate()
        if name not in existing_tables:
            existing_tables.append(name)
            self._group.attrs.update({"tables": existing_tables})


ImplementedTables().add_implementation(RoiTableV1)
ImplementedTables().add_implementation(MaskingRoiTableV1)
ImplementedTables().add_implementation(FeatureTableV1)
ImplementedTables().add_implementation(ConditionTableV1)

###################################################################################
#
# Utility functions to open and write tables
#
###################################################################################


def open_tables_container(
    store: StoreOrGroup,
    mode: AccessModeLiteral = "r+",
) -> TablesContainer:
    """Open a tables container from a Zarr store or group."""
    group = open_group_wrapper(store=store, mode=mode)
    return TablesContainer(group)


def open_table(
    store: StoreOrGroup,
    backend: TableBackend | None = None,
    mode: AccessModeLiteral = "r+",
) -> Table:
    """Open a table from a Zarr store or group."""
    group = open_group_wrapper(store=store, mode=mode)
    meta = _get_meta(group)
    return ImplementedTables().get_table(
        meta=meta, group=group, backend=backend, strict=False
    )


def open_table_as(
    store: StoreOrGroup,
    table_cls: type[TableType],
    backend: TableBackend | None = None,
    mode: AccessModeLiteral = "r+",
) -> TableType:
    """Open a table from a Zarr store or group as a specific type."""
    group = open_group_wrapper(store=store, mode=mode)
    return cast(
        "TableType",
        table_cls.from_group(
            group=group,
            backend=backend,
        ),
    )


def write_table(
    store: StoreOrGroup,
    table: Table,
    backend: TableBackend = DefaultTableBackend,
    mode: AccessModeLiteral = "a",
) -> None:
    """Write a table to a Zarr store or group.

    A table will be created at the given store location.

    Args:
        store (StoreOrGroup): The Zarr store or group to write the table to.
        table (Table): The table to write.
        backend (TableBackend): The backend to use for writing the table.
        mode (AccessModeLiteral): The access mode to use for the Zarr group.

    """
    group = open_group_wrapper(store=store, mode=mode)
    table.set_backend(
        group=group,
        backend=backend,
    )
    table.consolidate()
