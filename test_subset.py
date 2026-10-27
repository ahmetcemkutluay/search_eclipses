"""Build a small test subset of the catalogue from a target list.

Reads TIC IDs from a CSV target list, pulls the matching rows out of
records_v8.parquet, then pulls every measurement belonging to those
records out of measurements.parquet. Results go to ./test.

Schema, as of the catalogue version this was written against:
    records_v8.parquet    one row per star, keyed on 'id', TIC in 'tic' (string)
    measurements.parquet  long format, one row per measurement,
                          'record_id' -> records.id
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --- paths -----------------------------------------------------------------

CAT_DIR = Path("..") / "platolops2_catalog"
REC_PATH = CAT_DIR / "records_v8.parquet"
MEAS_PATH = CAT_DIR / "measurements.parquet"

TARGETS_PATH = Path("example_target_list.csv")
OUT_DIR = Path(".") / "test"


# --- helpers ---------------------------------------------------------------


def cast_to_column(values, path: Path, column: str) -> list:
    """Cast `values` to the type `column` actually has in the parquet file.

    A TIC read from CSV is an int; stored as a string in the catalogue the
    filter silently matches nothing, so the two are reconciled here rather
    than assumed.
    """
    field = pq.ParquetFile(path).schema_arrow.field(column)
    if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
        return [str(v) for v in values]
    return [int(v) for v in values]


def read_filtered(path: Path, column: str, values) -> pd.DataFrame:
    """Read only the rows where `column` is in `values`.

    The filter is pushed down to the parquet reader, so row groups that
    cannot contain a match are never loaded.
    """
    table = pq.read_table(path, filters=[(column, "in", list(values))])
    return table.to_pandas()


# --- main ------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(TARGETS_PATH)
    tics = targets["tic"].dropna().unique()
    print(f"{len(tics)} target TICs from {TARGETS_PATH.name}")

    # --- records -----------------------------------------------------------

    records = read_filtered(REC_PATH, "tic", cast_to_column(tics, REC_PATH, "tic"))
    print(f"records:      {len(records):>7} rows, "
          f"{records['tic'].nunique()}/{len(tics)} TICs matched")

    if records.empty:
        raise SystemExit("No targets found in the catalogue — check the TIC column.")

    # --- measurements ------------------------------------------------------

    record_ids = records["id"].dropna().unique()
    measurements = read_filtered(
        MEAS_PATH, "record_id", cast_to_column(record_ids, MEAS_PATH, "record_id")
    )
    print(f"measurements: {len(measurements):>7} rows for "
          f"{measurements['record_id'].nunique()} records")

    # --- write -------------------------------------------------------------

    records.to_parquet(OUT_DIR / "records_v8.parquet", index=False)
    measurements.to_parquet(OUT_DIR / "measurements.parquet", index=False)

    # CSV alongside, for eyeballing and for diffing in git. The pipeline
    # reads the parquet files; these are only for inspection.
    records.to_csv(OUT_DIR / "example_records_list.csv", index=False)
    measurements.to_csv(OUT_DIR / "example_measurements_list.csv", index=False)

    print(f"\nwritten to {OUT_DIR.resolve()}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name:<34} {f.stat().st_size / 1024:>8.1f} KB")

    # --- report ------------------------------------------------------------

    stored = set(records["tic"].astype(str))
    missing = [t for t in tics if str(t) not in stored]
    if missing:
        print(f"\n{len(missing)} TICs absent from the catalogue: {sorted(missing)}")

    if not measurements.empty:
        print("\nmeasurements per quantity:")
        counts = measurements["quantity"].value_counts()
        for quantity, n in counts.items():
            print(f"  {quantity:<28} {n:>6}")

        no_meas = set(records["id"]) - set(measurements["record_id"])
        if no_meas:
            print(f"\n{len(no_meas)} records have no measurements at all")


if __name__ == "__main__":
    main()