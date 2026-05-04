from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

import polars as pl
import zarr
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame

from ngio_tables._utils import NgioTablesError
from ngio_tables.backends._abstract_backend import AbstractTableBackend
from ngio_tables.backends._utils import (
    convert_polars_to_pandas,
    normalize_pandas_df,
    normalize_polars_lf,
)


class JsonTableBackend(AbstractTableBackend):
    """A class to load and write small tables in the zarr group .attrs (json)."""

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "json"

    @staticmethod
    def implements_anndata() -> bool:
        """Whether the handler implements the anndata protocol."""
        return False

    @staticmethod
    def implements_pandas() -> bool:
        """Whether the handler implements the dataframe protocol."""
        return True

    @staticmethod
    def implements_polars() -> bool:
        """Whether the handler implements the polars protocol."""
        return True

    def _get_table_group(self) -> zarr.Group:
        """Get the table group, creating it if it doesn't exist."""
        try:
            table_group = zarr.open_group(
                store=self._group.store,
                path=f"{self._group.path}/table",
                mode="a",
            )
        except Exception as e:
            raise NgioTablesError(
                "Could not get or create a 'table' group in the store "
                f"{self._group.store} path {self._group.path}/table."
            ) from e
        return table_group

    def load_as_polars_lf(self) -> LazyFrame:
        """Load the table as a polars LazyFrame."""
        table_dict = self._get_table_group().attrs.asdict()
        lf = pl.from_dict(cast("Mapping[str, Sequence[Any]]", table_dict)).lazy()
        return normalize_polars_lf(
            lf, index_key=self.index_key, index_type=self.index_type
        )

    def load_as_pandas_df(self) -> DataFrame:
        """Load the table as a pandas DataFrame."""
        lf = self.load_as_polars_lf()
        return convert_polars_to_pandas(
            lf, index_key=self.index_key, index_type=self.index_type
        )

    def load(self) -> LazyFrame:
        """Load the table as a polars LazyFrame."""
        return self.load_as_polars_lf()

    def _write_from_dict(self, table: dict) -> None:
        """Write the table from a dictionary to the store."""
        table_group = self._get_table_group()
        table_group.attrs.clear()
        table_group.attrs.update(table)

    def write_from_pandas(self, table: DataFrame) -> None:
        """Write the table from a pandas DataFrame."""
        table = normalize_pandas_df(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
            reset_index=True,
        )
        table_dict = table.to_dict(orient="list")
        self._write_from_dict(table=table_dict)

    def write_from_polars(self, table: PolarsDataFrame | LazyFrame) -> None:
        """Write the table from a polars DataFrame or LazyFrame."""
        table = normalize_polars_lf(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
        )
        if isinstance(table, LazyFrame):
            table = cast("PolarsDataFrame", table.collect())

        table_dict = table.to_dict(as_series=False)
        self._write_from_dict(table=table_dict)
