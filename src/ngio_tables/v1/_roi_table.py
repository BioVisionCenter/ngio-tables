"""Implementation of the ROI Table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

import logging
from collections.abc import Iterable
from typing import Any, ClassVar, Generic, Literal, Protocol, Self, TypeVar, cast
from uuid import uuid4

import polars as pl
import zarr
from polars import LazyFrame
from pydantic import BaseModel

from ngio_tables._abstract_table import (
    AbstractBaseTable,
    TabularData,
)
from ngio_tables._utils import (
    NgioTablesTableValidationError,
    NgioTablesValueError,
)
from ngio_tables.backends import (
    BackendMeta,
    TableBackend,
    convert_to_polars,
    normalize_polars_lf,
)

logger = logging.getLogger(f"ngio:{__name__}")

REQUIRED_COLUMNS = [
    "x_micrometer",
    "y_micrometer",
    "z_micrometer",
    "len_x_micrometer",
    "len_y_micrometer",
    "len_z_micrometer",
]

#####################
# Optional columns are not validated at the moment
# only a warning is raised if non optional columns are present
#####################

TIME_COLUMNS = [
    "t_second",
    "len_t_second",
]

ORIGIN_COLUMNS = [
    "x_micrometer_original",
    "y_micrometer_original",
    "z_micrometer_original",
]

TRANSLATION_COLUMNS = ["translation_x", "translation_y", "translation_z"]

PLATE_COLUMNS = [
    "plate_name",
    "row",
    "column",
    "path_in_well",
    "path_in_plate",
    "acquisition_id",
    "acquisition_name",
]

INDEX_COLUMNS = [
    "FieldIndex",
    "label",
]

OPTIONAL_COLUMNS = ORIGIN_COLUMNS + TRANSLATION_COLUMNS + PLATE_COLUMNS + INDEX_COLUMNS


class _RoiV1(BaseModel):
    """Class to represent a ROI in the ROI table."""

    name: str
    x_micrometer: float
    y_micrometer: float
    z_micrometer: float | None = None
    t_second: float | None = None
    len_x_micrometer: float
    len_y_micrometer: float
    len_z_micrometer: float | None = None
    len_t_second: float | None = None
    label: int | str | None = None
    extra_fields: dict[str, object] | None = None


class RoiAdapter(Protocol):
    """Protocol for ROI adapters."""

    name: str
    label: int | str | None

    @classmethod
    def from_roi_v1(cls, roi: _RoiV1) -> Self:
        """Convert the input RoiV1 object to the adapter's format."""
        ...

    def to_roi_v1(self) -> _RoiV1:
        """Convert the object to a RoiV1 object."""
        ...


class RoiV1(_RoiV1):
    """Class to represent a ROI in the ROI table."""

    def to_roi_v1(self) -> _RoiV1:
        """Convert the object to a RoiV1 object."""
        return self

    @classmethod
    def from_roi_v1(cls, roi: _RoiV1) -> "RoiV1":
        """Convert the input RoiV1 object to the adapter's format."""
        return cls(**roi.model_dump())


RoiType = TypeVar("RoiType", bound=RoiAdapter)


def _check_optional_columns(col_name: str) -> None:
    """Check if the column name is in the optional columns."""
    if col_name not in OPTIONAL_COLUMNS + TIME_COLUMNS:
        logger.warning(f"Column {col_name} is not in the optional columns.")


def _dataframe_to_rois(
    dataframe: pl.DataFrame,
    roi_type: type[RoiType],
    index_key: str | None,
    required_columns: list[str] = REQUIRED_COLUMNS,
) -> dict[str, RoiType]:
    """Convert a DataFrame to a dict of RoiV1 objects."""
    _missing_columns = set(required_columns).difference(set(dataframe.columns))
    if len(_missing_columns) != 0:
        raise NgioTablesTableValidationError(
            f"Could not find required columns: {_missing_columns} in the table."
        )

    # Exclude index_key: polars keeps it as a regular column (unlike pandas index)
    _index_cols = [index_key] if index_key else []
    all_known_cols = required_columns + TIME_COLUMNS + ["label"] + _index_cols
    extra_columns = set(dataframe.columns).difference(set(all_known_cols))

    for col in extra_columns:
        _check_optional_columns(col)

    label_is_index = index_key == "label"

    rois: dict[str, RoiType] = {}
    for row in dataframe.iter_rows(named=True):
        extras = {col: row.get(col) for col in extra_columns}

        if label_is_index:
            label = int(row[index_key])  # type: ignore[index]
        else:
            label = row.get("label")

        roi = _RoiV1(
            name=str(row[index_key]) if index_key is not None else str(row),
            x_micrometer=row["x_micrometer"],
            y_micrometer=row["y_micrometer"],
            z_micrometer=row.get("z_micrometer"),
            t_second=row.get("t_second"),
            len_x_micrometer=row["len_x_micrometer"],
            len_y_micrometer=row["len_y_micrometer"],
            len_z_micrometer=row.get("len_z_micrometer"),
            len_t_second=row.get("len_t_second"),
            label=label,
            extra_fields=extras if extras else None,
        )
        rois[roi.name] = roi_type.from_roi_v1(roi)
    return rois


def _rois_to_dataframe(rois: dict[str, RoiType], index_key: str | None) -> pl.DataFrame:
    """Convert a dict of RoiV1 objects to a DataFrame."""
    data = []
    for roi in rois.values():
        roi = roi.to_roi_v1()
        len_z_um = roi.len_z_micrometer if roi.len_z_micrometer is not None else 1.0
        row: dict[str, Any] = {
            cast("str", index_key): roi.name,
            "x_micrometer": roi.x_micrometer,
            "y_micrometer": roi.y_micrometer,
            "z_micrometer": roi.z_micrometer if roi.z_micrometer is not None else 0.0,
            "len_x_micrometer": roi.len_x_micrometer,
            "len_y_micrometer": roi.len_y_micrometer,
            "len_z_micrometer": len_z_um,
        }

        if roi.t_second is not None or roi.len_t_second is not None:
            len_t_sec = roi.len_t_second if roi.len_t_second is not None else 1.0
            row["t_second"] = roi.t_second if roi.t_second is not None else 0.0
            row["len_t_second"] = len_t_sec

        if roi.label is not None and index_key != "label":
            row["label"] = roi.label

        if roi.extra_fields:
            for col, val in roi.extra_fields.items():
                _check_optional_columns(col)
                row[col] = cast("str | int | float", val)
        data.append(row)

    lf = pl.DataFrame(data).lazy()
    lf = normalize_polars_lf(lf, index_key=index_key)
    return cast("pl.DataFrame", lf.collect())


class RoiDictWrapper(Generic[RoiType]):
    """A wrapper for a dictionary of ROIs to provide a consistent interface."""

    def __init__(self, rois: Iterable[RoiType]) -> None:
        """Initialize the RoiDictWrapper."""
        self._rois_by_name: dict[str, RoiType] = {}
        self._rois_by_label: dict[int | str, RoiType] = {}
        for roi in rois:
            name = roi.name
            if name in self._rois_by_name:
                name = f"{name}_{uuid4().hex[:8]}"
            self._rois_by_name[name] = roi
            if roi.label is not None:
                self._rois_by_label[roi.label] = roi

    def get_by_name(self, name: str, default: RoiType | None = None) -> RoiType | None:
        """Get an ROI by its name."""
        return self._rois_by_name.get(name, default)

    def get_by_label(
        self, label: int, default: RoiType | None = None
    ) -> RoiType | None:
        """Get an ROI by its label."""
        return self._rois_by_label.get(label, default)

    def _add_roi(self, roi: RoiType, overwrite: bool = False) -> None:
        """Add an ROI to the wrapper."""
        if roi.name in self._rois_by_name and not overwrite:
            raise NgioTablesValueError(f"ROI with name {roi.name} already exists.")

        self._rois_by_name[roi.name] = roi
        if roi.label is not None:
            self._rois_by_label[roi.label] = roi

    def add_rois(
        self, rois: RoiType | Iterable[RoiType], overwrite: bool = False
    ) -> None:
        """Add ROIs to the wrapper."""
        _rois: Iterable[RoiType]
        if isinstance(rois, BaseModel) or not isinstance(rois, Iterable):
            _rois = cast("Iterable[RoiType]", [rois])
        else:
            _rois = cast("Iterable[RoiType]", rois)
        for _roi in _rois:
            self._add_roi(_roi, overwrite=overwrite)

    def to_list(self) -> list[RoiType]:
        """Return the list of ROIs."""
        return list(self._rois_by_name.values())

    def to_dataframe(self, index_key: str | None = None) -> pl.DataFrame:
        """Convert the ROIs to a DataFrame."""
        return _rois_to_dataframe(self._rois_by_name, index_key=index_key)

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pl.DataFrame,
        roi_type: type[RoiType],
        index_key: str | None = None,
        required_columns: list[str] = REQUIRED_COLUMNS,
    ) -> "RoiDictWrapper":
        """Create a RoiDictWrapper from a DataFrame."""
        rois = _dataframe_to_rois(
            dataframe,
            roi_type=roi_type,
            index_key=index_key,
            required_columns=required_columns,
        )
        return cls(rois.values())


def _table_to_rois(
    table: TabularData,
    roi_type: type[RoiType],
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
    required_columns: list[str] = REQUIRED_COLUMNS,
) -> tuple[LazyFrame, RoiDictWrapper[RoiType]]:
    """Convert a table to a dictionary of ROIs.

    Args:
        table: The table to convert.
        roi_type: The type of the ROIs in the table.
        index_key: The column name to use as the index of the DataFrame.
        index_type: The type of the index column in the DataFrame.
        required_columns: The required columns in the DataFrame.

    Returns:
        A tuple containing the LazyFrame and a RoiDictWrapper with the ROIs.
    """
    lf = convert_to_polars(
        table,
        index_key=index_key,
        index_type=index_type,
    )
    df = cast("pl.DataFrame", lf.collect())
    roi_dict_wrapper = RoiDictWrapper.from_dataframe(
        df, roi_type=roi_type, index_key=index_key, required_columns=required_columns
    )
    return lf, roi_dict_wrapper


class _RoiTableBase(AbstractBaseTable, Generic[RoiType]):
    """Base ROI table v1 implementation shared by all ROI table variants."""

    _rois: RoiDictWrapper[RoiType]
    _roi_cls: ClassVar[type[RoiAdapter]]

    def __init_subclass__(
        cls, roi_type: type[RoiAdapter] | None = None, **kwargs: Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        if roi_type is not None:
            cls._roi_cls = roi_type

    def __init__(
        self,
        *,
        rois: Iterable[RoiType] | None = None,
        meta: BackendMeta,
        required_columns: list[str] = REQUIRED_COLUMNS,
    ) -> None:
        """Initialize the _RoiTableBase."""
        rois = rois or []
        self._rois: RoiDictWrapper[RoiType] = RoiDictWrapper(rois)
        table = self._rois.to_dataframe(index_key=meta.index_key) if rois else None

        self._required_columns = required_columns
        super().__init__(table_data=table, meta=meta)

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        rois = self.rois()
        prop = f"num_rois={len(rois)}"
        class_name = self.__class__.__name__
        return f"{class_name}({prop})"

    @staticmethod
    def table_type() -> str:
        """Return the type of the table."""
        return "generic_roi_table"

    @staticmethod
    def version() -> Literal["1"]:
        """Return the version of the fractal table."""
        return "1"

    @property
    def table_data(self) -> TabularData:
        """Return the table."""
        if self._rois is None:
            return super().table_data

        if len(self.rois()) > 0:
            self._table_data = self._rois.to_dataframe(index_key=self.meta.index_key)
        return super().table_data

    def set_table_data(
        self, table_data: TabularData | None = None, refresh: bool = False
    ) -> None:
        """Set the table data."""
        if table_data is not None:
            if not isinstance(table_data, TabularData):
                raise NgioTablesValueError(
                    "The table must be a pandas DataFrame, polars LazyFrame, "
                    " or AnnData object."
                )

            table_data, rois = _table_to_rois(
                table_data,
                roi_type=self.__class__._roi_cls,
                index_key=self.index_key,
                index_type=self.index_type,
                required_columns=REQUIRED_COLUMNS,
            )
            self._table_data = table_data
            self._rois = rois
            return None

        if self._table_data is not None and not refresh:
            return None

        if self._table_backend is None:
            raise NgioTablesValueError(
                "The table does not have a DataFrame in memory nor a backend."
            )

        table_data, rois = _table_to_rois(
            self._table_backend.load(),
            roi_type=self.__class__._roi_cls,
            index_key=self.index_key,
            index_type=self.index_type,
            required_columns=REQUIRED_COLUMNS,
        )
        self._table_data = table_data
        self._rois = rois

    def rois(self) -> list[RoiType]:
        """List all ROIs in the table."""
        if not self._rois.to_list() and self._table_backend is not None:
            self.set_table_data()
        return self._rois.to_list()

    def add(self, roi: RoiType | Iterable[RoiType], overwrite: bool = False) -> None:
        """Append ROIs to the current table.

        Args:
            roi: A single ROI or a list of ROIs to add to the table.
            overwrite: If True, overwrite existing ROIs with the same name.
        """
        self._rois.add_rois(roi, overwrite=overwrite)

    def get(self, roi_name: str) -> RoiType:
        """Get an ROI from the table."""
        roi = self._rois.get_by_name(roi_name)
        if roi is None:
            raise NgioTablesValueError(
                f"ROI with name {roi_name} not found in the table."
            )
        return roi

    @classmethod
    def from_table_data(
        cls, table_data: TabularData, meta: BackendMeta
    ) -> "_RoiTableBase":
        """Create a new ROI table from a table data."""
        _, rois = _table_to_rois(
            table=table_data,
            roi_type=cls._roi_cls,
            index_key=meta.index_key,
            index_type=meta.index_type,
            required_columns=REQUIRED_COLUMNS,
        )
        return cls(rois=rois.to_list(), meta=meta)

    @classmethod
    def from_group(
        cls,
        group: zarr.Group,
        backend: TableBackend | None = None,
    ) -> "_RoiTableBase":
        """Create a new _RoiTableBase from a zarr group."""
        return cls._from_group(
            group=group,
            backend=backend,
            meta_model=BackendMeta,
        )


class RoiTableV1Meta(BackendMeta):
    """Metadata for the ROI table."""

    table_version: Literal["1"] = "1"
    type: Literal["roi_table"] = "roi_table"
    index_key: str | None = "FieldIndex"
    index_type: Literal["str", "int"] | None = "str"


class AbstractRoiTableV1(_RoiTableBase[RoiType]):
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(
        self,
        rois: Iterable[RoiType] | None = None,
        *,
        meta: RoiTableV1Meta | None = None,
    ) -> None:
        """Create a new ROI table."""
        if meta is None:
            meta = RoiTableV1Meta()

        if meta.index_key is None:
            meta.index_key = "FieldIndex"

        if meta.index_type is None:
            meta.index_type = "str"
        super().__init__(meta=meta, rois=rois)

    @classmethod
    def from_group(
        cls,
        group: zarr.Group,
        backend: TableBackend | None = None,
    ) -> "AbstractRoiTableV1":
        """Create a new RoiTableV1 from a zarr group."""
        table = cls._from_group(
            group=group,
            backend=backend,
            meta_model=RoiTableV1Meta,
        )
        return table

    @staticmethod
    def table_type() -> Literal["roi_table"]:
        """Return the type of the table."""
        return "roi_table"


class RegionMeta(BaseModel):
    """Metadata for the region."""

    path: str


class MaskingRoiTableV1Meta(BackendMeta):
    """Metadata for the ROI table."""

    table_version: Literal["1"] = "1"
    type: Literal["masking_roi_table"] = "masking_roi_table"
    region: RegionMeta | None = None
    instance_key: str = "label"
    index_key: str | None = "label"
    index_type: Literal["int", "str"] | None = "int"


class AbstractMaskingRoiTableV1(_RoiTableBase[RoiType]):
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(
        self,
        rois: Iterable[RoiType] | None = None,
        *,
        reference_label: str | None = None,
        meta: MaskingRoiTableV1Meta | None = None,
    ) -> None:
        """Create a new ROI table."""
        if meta is None:
            meta = MaskingRoiTableV1Meta()

        if reference_label is not None:
            path = f"../labels/{reference_label}"
            meta.region = RegionMeta(path=path)

        if meta.index_key is None:
            meta.index_key = "label"

        if meta.index_type is None:
            meta.index_type = "int"
        meta.instance_key = meta.index_key
        super().__init__(meta=meta, rois=rois)

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        rois = self.rois()
        if self.reference_label is not None:
            prop = f"num_rois={len(rois)}, reference_label={self.reference_label}"
        else:
            prop = f"num_rois={len(rois)}"
        return f"MaskingRoiTableV1({prop})"

    @classmethod
    def from_group(
        cls,
        group: zarr.Group,
        backend: TableBackend | None = None,
    ) -> "AbstractMaskingRoiTableV1":
        """Create a new MaskingRoiTableV1 from a zarr group."""
        table = cls._from_group(
            group=group,
            backend=backend,
            meta_model=MaskingRoiTableV1Meta,
        )
        return table

    @staticmethod
    def table_type() -> Literal["masking_roi_table"]:
        """Return the type of the table."""
        return "masking_roi_table"

    @property
    def meta(self) -> MaskingRoiTableV1Meta:
        """Return the metadata of the table."""
        if not isinstance(self._meta, MaskingRoiTableV1Meta):
            raise NgioTablesValueError(
                "The metadata of the table is not of type MaskingRoiTableV1Meta."
            )
        return self._meta

    @property
    def reference_label(self) -> str | None:
        """Return the reference label."""
        path = self.meta.region
        if path is None:
            return None

        path = path.path
        path = path.split("/")[-1]
        return path

    def get_label(self, label: int) -> RoiType:
        """Get an ROI by label."""
        self._rois = self._rois
        roi = self._rois.get_by_label(label)
        if roi is None:
            raise NgioTablesValueError(f"ROI with label {label} not found.")
        return roi


class RoiTableV1(AbstractRoiTableV1[RoiV1], roi_type=RoiV1):
    """Class to handle fractal ROI tables with ROIs of type RoiV1."""


class MaskingRoiTableV1(AbstractMaskingRoiTableV1[RoiV1], roi_type=RoiV1):
    """Class to handle fractal masking ROI tables with ROIs of type RoiV1."""
