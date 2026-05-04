from pathlib import Path

import pandas as pd
import pytest

from ngio_tables._tables_container import RoiTable, open_table, write_table
from ngio_tables._utils import NgioTablesValueError, open_group_wrapper
from ngio_tables.backends import AnnDataBackend
from ngio_tables.v1 import AbstractRoiTableV1, RoiTableV1
from ngio_tables.v1._roi_table import RoiTableV1Meta, RoiV1


def test_roi_table_v1(tmp_path: Path):
    rois = [
        RoiV1(
            name="roi1",
            x_micrometer=0.0,
            y_micrometer=0.0,
            z_micrometer=0.0,
            len_x_micrometer=1.0,
            len_y_micrometer=1.0,
            len_z_micrometer=1.0,
        )
    ]

    table = RoiTableV1(rois=rois)
    assert isinstance(table.__repr__(), str)

    table.add(
        roi=RoiV1(
            name="roi2",
            x_micrometer=0.0,
            y_micrometer=0.0,
            z_micrometer=0.0,
            len_x_micrometer=1.0,
            len_y_micrometer=1.0,
            len_z_micrometer=1.0,
        )
    )

    with pytest.raises(NgioTablesValueError):
        table.add(
            roi=RoiV1(
                name="roi2",
                x_micrometer=0.0,
                y_micrometer=0.0,
                z_micrometer=0.0,
                len_x_micrometer=1.0,
                len_y_micrometer=1.0,
                len_z_micrometer=1.0,
            )
        )

    table.add(
        roi=RoiV1(
            name="roi2",
            x_micrometer=0.0,
            y_micrometer=0.0,
            z_micrometer=0.0,
            len_x_micrometer=1.0,
            len_y_micrometer=1.0,
            len_z_micrometer=1.0,
        ),
        overwrite=True,
    )
    assert len(table.rois()) == 2
    write_table(store=tmp_path / "roi_table.zarr", table=table, backend="anndata")

    loaded_table = open_table(store=tmp_path / "roi_table.zarr")
    assert isinstance(loaded_table, RoiTableV1)
    assert len(loaded_table.rois()) == 2
    assert loaded_table.get("roi1") == table.get("roi1")
    assert loaded_table.get("roi2") == table.get("roi2")

    with pytest.raises(NgioTablesValueError):
        loaded_table.get("roi3")

    assert loaded_table.meta.backend == "anndata"
    meta_dict = loaded_table._meta.model_dump()
    assert meta_dict.get("table_version") == loaded_table.version()
    assert meta_dict.get("type") == loaded_table.table_type()


def test_custom_roi_type_via_keyword():
    class CustomRoiTable(AbstractRoiTableV1[RoiV1], roi_type=RoiV1):
        pass

    assert CustomRoiTable._roi_cls is RoiV1
    table = CustomRoiTable()
    assert table.__class__._roi_cls is RoiV1


def test_roi_no_index(tmp_path: Path):
    """ngio needs to support reading a table without an index. for legacy reasons"""
    group = open_group_wrapper(tmp_path / "roi_table.zarr", mode="a")
    backend = AnnDataBackend()
    backend.set_group(group)

    roi_table = pd.DataFrame(
        {
            "x_micrometer": [0.0, 1.0],
            "y_micrometer": [0.0, 1.0],
            "z_micrometer": [0.0, 1.0],
            "len_x_micrometer": [1.0, 1.0],
            "len_y_micrometer": [1.0, 1.0],
            "len_z_micrometer": [1.0, 1.0],
        }
    )
    roi_table.index = pd.Index(["roi_1", "roi_2"])

    backend.write(
        roi_table,
        metadata=RoiTableV1Meta().model_dump(exclude_none=True),
    )

    roi_table = RoiTable.from_group(group=group)
    assert isinstance(roi_table, RoiTable)
    assert len(roi_table.rois()) == 2
