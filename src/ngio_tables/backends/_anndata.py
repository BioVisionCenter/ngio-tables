import zarr
from anndata import AnnData
from anndata._settings import settings
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame
from zarr.storage import FsspecStore, LocalStore, MemoryStore

from ngio_tables._utils import NgioTablesValueError, copy_group
from ngio_tables.backends._abstract_backend import AbstractTableBackend
from ngio_tables.backends._anndata_utils import (
    custom_anndata_read_zarr,
)
from ngio_tables.backends._utils import (
    convert_pandas_to_anndata,
    convert_polars_to_anndata,
    normalize_anndata,
)


class AnnDataBackend(AbstractTableBackend):
    """A class to load and write tables from/to an AnnData object."""

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "anndata"

    @staticmethod
    def implements_anndata() -> bool:
        """Check if the backend implements the anndata protocol."""
        return True

    @staticmethod
    def implements_pandas() -> bool:
        """Whether the handler implements the dataframe protocol."""
        return True

    @staticmethod
    def implements_polars() -> bool:
        """Whether the handler implements the polars protocol."""
        return True

    def load_as_anndata(self) -> AnnData:
        """Load the table as an AnnData object."""
        settings.zarr_write_format = self._group.metadata.zarr_format
        anndata = custom_anndata_read_zarr(self._group)
        anndata = normalize_anndata(anndata, index_key=self.index_key)
        return anndata

    def load(self) -> AnnData:
        """Load the table as an AnnData object."""
        return self.load_as_anndata()

    def _write_to_local_store(
        self, store: LocalStore, path: str, table: AnnData
    ) -> None:
        """Write the AnnData table to a LocalStore."""
        store_path = f"{store.root}/{path}"
        table.write_zarr(store_path)  # type: ignore  # ty false positive: AnnData write_zarr stubs are ambiguous

    def _write_to_fsspec_store(
        self, store: FsspecStore, path: str, table: AnnData
    ) -> None:
        """Write the AnnData table to a FsspecStore."""
        full_url = f"{store.path}/{path}"
        fs = store.fs
        mapper = fs.get_mapper(full_url)
        table.write_zarr(mapper)  # type: ignore  # ty false positive: AnnData write_zarr stubs are ambiguous

    def _write_to_memory_store(
        self, store: MemoryStore, path: str, table: AnnData
    ) -> None:
        """Write the AnnData table to a MemoryStore."""
        store = MemoryStore()
        table.write_zarr(store)  # type: ignore  # ty false positive: AnnData write_zarr stubs are ambiguous
        anndata_group = zarr.open_group(store, mode="r")
        copy_group(
            anndata_group,
            self._group,
            suppress_warnings=True,
        )

    def _cleanup_after_write(self) -> None:
        """Clean up any temporary data after writing."""
        fresh_group = zarr.open_group(
            store=self._group.store,
            path=self._group.path,
            mode="r+",
        )
        try:
            raw_group = fresh_group["raw"]
        except KeyError:
            return
        # Remove "raw" entry (encoding-type "null") that anndata <0.11 can't read
        if dict(raw_group.attrs).get("encoding-type") == "null":
            del fresh_group["raw"]

    def write_from_anndata(self, table: AnnData) -> None:
        """Serialize the table from an AnnData object."""
        # Make sure to use the correct zarr format
        settings.zarr_write_format = self._group.metadata.zarr_format
        store = self._group.store
        path = self._group.path
        if isinstance(store, LocalStore):
            self._write_to_local_store(
                store,
                path,
                table,
            )
        elif isinstance(store, FsspecStore):
            self._write_to_fsspec_store(
                store,
                path,
                table,
            )
        elif isinstance(store, MemoryStore):
            self._write_to_memory_store(
                store,
                path,
                table,
            )
        else:
            raise NgioTablesValueError(
                f"Ngio does not support writing an AnnData table to a "
                f"store of type {type(store)}. "
                "Please make sure to use a compatible "
                "store like a LocalStore, or FsspecStore."
            )
        self._cleanup_after_write()

    def write_from_pandas(self, table: DataFrame) -> None:
        """Serialize the table from a pandas DataFrame."""
        anndata = convert_pandas_to_anndata(
            table,
            index_key=self.index_key,
        )
        self.write_from_anndata(anndata)

    def write_from_polars(self, table: PolarsDataFrame | LazyFrame) -> None:
        """Consolidate the metadata in the store."""
        anndata = convert_polars_to_anndata(
            table,
            index_key=self.index_key,
        )
        self.write_from_anndata(anndata)


class AnnDataBackendV1(AnnDataBackend):
    """A wrapper for the AnnData backend that for backwards compatibility."""

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "anndata_v1"
