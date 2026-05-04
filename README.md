# ngio-tables

Standalone OME-Zarr table handling extracted from [ngio](https://github.com/BioVisionCenter/ngio).

This package provides read/write support for tabular data in OME-Zarr files (ROI tables, feature tables, condition tables) without depending on the full ngio package.

## Installation

```bash
pip install ngio-tables
```

## Usage

```python
from ngio_tables import (
    RoiTable, FeatureTable, ConditionTable,
    TablesContainer, open_table, write_table,
)
from ngio_tables._utils import Roi

# Create and write a ROI table
rois = [Roi.from_values(name="roi1", slices={"x": (0, 100), "y": (0, 100), "z": (0, 10)})]
table = RoiTable(rois=rois)
write_table(store="path/to/table.zarr", table=table)

# Read it back
loaded = open_table(store="path/to/table.zarr")
print(loaded.rois())
```

## Relationship to ngio

`ngio-tables` is a subset of [ngio](https://github.com/BioVisionCenter/ngio). It contains only the table-handling code and the minimal utilities it depends on. The public API is identical to `ngio.tables`.
