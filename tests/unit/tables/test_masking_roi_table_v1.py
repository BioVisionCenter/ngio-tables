from pathlib import Path

import pytest

from ngio_tables._tables_container import open_table, write_table
from ngio_tables._utils import NgioTablesValueError
from ngio_tables.v1._roi_table import MaskingRoiTableV1, RoiV1


def test_masking_roi_table_v1(tmp_path: Path):
    rois = {
        1: RoiV1(
            name="1",
            x_micrometer=0.0,
            y_micrometer=0.0,
            z_micrometer=0.0,
            len_x_micrometer=10.0,
            len_y_micrometer=10.0,
            len_z_micrometer=5.0,
        )
    }

    table = MaskingRoiTableV1(rois=rois.values(), reference_label="label")
    assert isinstance(table.__repr__(), str)
    assert table.reference_label == "label"
    assert table.meta.region is not None
    assert table.meta.region.path == "../labels/label"

    table.add(
        roi=RoiV1(
            name="2",
            x_micrometer=0.0,
            y_micrometer=0.0,
            z_micrometer=0.0,
            len_x_micrometer=10.0,
            len_y_micrometer=10.0,
            len_z_micrometer=5.0,
        )
    )

    with pytest.raises(NgioTablesValueError):
        table.add(
            roi=RoiV1(
                name="2",
                x_micrometer=0.0,
                y_micrometer=0.0,
                z_micrometer=0.0,
                len_x_micrometer=10.0,
                len_y_micrometer=10.0,
                len_z_micrometer=5.0,
            )
        )

    write_table(store=tmp_path / "roi_table.zarr", table=table, backend="anndata")

    loaded_table = open_table(store=tmp_path / "roi_table.zarr")
    assert isinstance(loaded_table, MaskingRoiTableV1)

    assert loaded_table.meta.backend == "anndata"
    meta_dict = loaded_table._meta.model_dump()
    assert meta_dict.get("table_version") == loaded_table.version()
    assert meta_dict.get("type") == loaded_table.table_type()
