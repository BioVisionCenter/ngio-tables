"""Implementation of a generic table class."""

import zarr

from ngio_tables._abstract_table import AbstractBaseTable
from ngio_tables.backends import BackendMeta, TableBackend


class GenericTableMeta(BackendMeta):
    """Metadata for the generic table.

    This is used to store metadata for a generic table.
    It does not have a specific definition.
    """

    table_version: str | None = "1"
    type: str | None = "generic_table"


class GenericTable(AbstractBaseTable):
    """Class to a non-specific table.

    This can be used to load any table that does not have
    a specific definition.
    """

    @staticmethod
    def table_type() -> str:
        """Return the type of the table."""
        return "generic_table"

    @staticmethod
    def version() -> str:
        """The generic table does not have a version.

        Since does not follow a specific schema.
        """
        return "1"

    @classmethod
    def from_group(
        cls,
        group: zarr.Group,
        backend: TableBackend | None = None,
    ) -> "GenericTable":
        """Create a new GenericTable from a zarr group."""
        return cls._from_group(
            group=group,
            backend=backend,
            meta_model=BackendMeta,
        )
