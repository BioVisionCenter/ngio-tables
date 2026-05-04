"""Implementation of a generic table class."""

import builtins
from abc import ABC, abstractmethod
from typing import Any, Literal, Self

import pandas as pd
import polars as pl
import zarr
from anndata import AnnData

from ngio_tables._utils import NgioTablesValueError
from ngio_tables.backends import (
    BackendMeta,
    DefaultTableBackend,
    ImplementedTableBackends,
    TableBackend,
    TableBackendProtocol,
    TabularData,
    convert_to_anndata,
    convert_to_pandas,
    convert_to_polars,
    normalize_table,
)


class AbstractBaseTable(ABC):
    """Abstract base class for a table.

    This is used to define common methods and properties
    for all tables.

    This class is not meant to be used directly.
    """

    def __init__(
        self,
        table_data: TabularData | None = None,
        *,
        meta: BackendMeta | None = None,
    ) -> None:
        """Initialize the table."""
        if meta is None:
            meta = BackendMeta()

        self._meta = meta
        if table_data is not None:
            table_data = normalize_table(
                table_data,
                index_key=meta.index_key,
                index_type=meta.index_type,
            )
        self._table_data = table_data
        self._table_backend = None

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        return f"{self.__class__.__name__}"

    @staticmethod
    @abstractmethod
    def table_type() -> str:
        """Return the type of the table."""
        ...

    @staticmethod
    @abstractmethod
    def version() -> str:
        """The generic table does not have a version.

        Since does not follow a specific schema.
        """
        ...

    @property
    def backend_name(self) -> str | None:
        """Return the name of the backend."""
        if self._table_backend is None:
            return None
        return self._table_backend.backend_name()

    @property
    def meta(self) -> BackendMeta:
        """Return the metadata of the table."""
        return self._meta

    @property
    def index_key(self) -> str | None:
        """Get the index key."""
        return self._meta.index_key

    @property
    def index_type(self) -> Literal["int", "str"] | None:
        """Get the index type."""
        return self._meta.index_type

    def load_as_anndata(self) -> AnnData:
        """Load the table as an AnnData object."""
        if self._table_backend is None:
            raise NgioTablesValueError("No backend set for the table.")
        return self._table_backend.load_as_anndata()

    def load_as_pandas_df(self) -> pd.DataFrame:
        """Load the table as a pandas DataFrame."""
        if self._table_backend is None:
            raise NgioTablesValueError("No backend set for the table.")
        return self._table_backend.load_as_pandas_df()

    def load_as_polars_lf(self) -> pl.LazyFrame:
        """Load the table as a polars LazyFrame."""
        if self._table_backend is None:
            raise NgioTablesValueError("No backend set for the table.")
        return self._table_backend.load_as_polars_lf()

    @property
    def table_data(self) -> TabularData:
        """Return the table."""
        if self._table_data is not None:
            return self._table_data

        if self._table_backend is None:
            raise NgioTablesValueError(
                "The table does not have a DataFrame in memory nor a backend."
            )

        self._table_data = self._table_backend.load()
        return self._table_data

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the table as a DataFrame."""
        return convert_to_pandas(
            self.table_data, index_key=self.index_key, index_type=self.index_type
        )

    @property
    def lazy_frame(self) -> pl.LazyFrame:
        """Return the table as a LazyFrame."""
        return convert_to_polars(
            self.table_data, index_key=self.index_key, index_type=self.index_type
        )

    @property
    def anndata(self) -> AnnData:
        """Return the table as an AnnData object."""
        return convert_to_anndata(self.table_data, index_key=self.index_key)

    @staticmethod
    def _load_backend(
        meta: BackendMeta,
        group: zarr.Group,
        backend: TableBackend,
    ) -> TableBackendProtocol:
        """Create a new table backend attached to the given zarr group."""
        if isinstance(backend, str):
            return ImplementedTableBackends().get_backend(
                backend_name=backend,
                group=group,
                index_key=meta.index_key,
                index_type=meta.index_type,
            )
        backend.set_group(
            group=group,
            index_key=meta.index_key,
            index_type=meta.index_type,
        )
        return backend

    def set_table_data(
        self,
        table_data: TabularData | None = None,
        refresh: bool = False,
    ) -> None:
        """Set the table.

        If an object is passed, it will be used as the table.
        If None is passed, the table will be loaded from the backend.

        If refresh is True, the table will be reloaded from the backend.
            If table is not None, this will be ignored.
        """
        if table_data is not None:
            if not isinstance(table_data, TabularData):
                raise NgioTablesValueError(
                    "The table must be a pandas DataFrame, polars LazyFrame, "
                    " or AnnData object."
                )

            self._table_data = normalize_table(
                table_data,
                index_key=self.index_key,
                index_type=self.index_type,
            )
            return None

        if self._table_data is not None and not refresh:
            return None

        if self._table_backend is None:
            raise NgioTablesValueError(
                "The table does not have a DataFrame in memory nor a backend."
            )
        self._table_data = self._table_backend.load()

    def set_backend(
        self,
        group: zarr.Group | None = None,
        backend: TableBackend = DefaultTableBackend,
    ) -> None:
        """Set the backend of the table."""
        if group is None:
            if self._table_backend is None:
                raise NgioTablesValueError(
                    "No backend set for the table yet. A zarr.Group must be provided."
                )
            group = self._table_backend.group

        meta = self._meta
        _backend = self._load_backend(
            meta=meta,
            group=group,
            backend=backend,
        )
        self._table_backend = _backend

    @classmethod
    def _from_group(
        cls,
        group: zarr.Group,
        meta_model: builtins.type[BackendMeta],
        backend: TableBackend | None = None,
    ) -> Self:
        """Create a new table from a zarr group."""
        attrs: dict[str, Any] = group.attrs.asdict()
        meta = meta_model(**attrs)
        table = cls(meta=meta)
        if backend is None:
            backend = meta.backend
        table.set_backend(group=group, backend=backend)
        return table

    @classmethod
    @abstractmethod
    def from_group(
        cls,
        group: zarr.Group,
        backend: TableBackend | None = None,
    ) -> Self:
        """Create a new table from a zarr group."""
        pass

    @classmethod
    def from_table_data(cls, table_data: TabularData, meta: BackendMeta) -> Self:
        """Create a new ROI table from a Zarr group handler."""
        return cls(
            table_data=table_data,
            meta=meta,
        )

    def consolidate(self) -> None:
        """Write the current state of the table to the Zarr file."""
        if self._table_backend is None:
            raise NgioTablesValueError(
                "No backend set for the table. "
                "Please add the table to a OME-Zarr Image before calling consolidate."
            )

        self._table_backend.write(
            self.table_data,
            metadata=self._meta.model_dump(exclude_none=True),
        )
