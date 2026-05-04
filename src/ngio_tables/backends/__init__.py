"""Ngio Tables backend implementations."""

from ngio_tables.backends._abstract_backend import AbstractTableBackend, BackendMeta
from ngio_tables.backends._anndata import AnnDataBackend, AnnDataBackendV1
from ngio_tables.backends._csv import CsvTableBackend
from ngio_tables.backends._json import JsonTableBackend
from ngio_tables.backends._parquet import ParquetTableBackend
from ngio_tables.backends._table_backends import (
    DefaultTableBackend,
    ImplementedTableBackends,
    TableBackend,
    TableBackendProtocol,
)
from ngio_tables.backends._utils import (
    TabularData,
    convert_anndata_to_pandas,
    convert_anndata_to_polars,
    convert_pandas_to_anndata,
    convert_pandas_to_polars,
    convert_polars_to_anndata,
    convert_polars_to_pandas,
    convert_to_anndata,
    convert_to_pandas,
    convert_to_polars,
    normalize_anndata,
    normalize_pandas_df,
    normalize_polars_lf,
    normalize_table,
)

__all__ = [
    "AbstractTableBackend",
    "AnnDataBackend",
    "AnnDataBackendV1",
    "BackendMeta",
    "CsvTableBackend",
    "DefaultTableBackend",
    "ImplementedTableBackends",
    "JsonTableBackend",
    "ParquetTableBackend",
    "TableBackend",
    "TableBackendProtocol",
    "TabularData",
    "convert_anndata_to_pandas",
    "convert_anndata_to_polars",
    "convert_pandas_to_anndata",
    "convert_pandas_to_polars",
    "convert_polars_to_anndata",
    "convert_polars_to_pandas",
    "convert_to_anndata",
    "convert_to_pandas",
    "convert_to_polars",
    "normalize_anndata",
    "normalize_pandas_df",
    "normalize_polars_lf",
    "normalize_table",
]
