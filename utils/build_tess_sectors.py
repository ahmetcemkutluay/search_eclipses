"""
Generates/updates the `tess_sectors.dat` file from your own TESS FITS files.

Rationale: The `get_tess_sectors()` function in ECLIPSR relies on the JD (Julian Date) boundaries
defined in the `eclipsr/data/tess_sectors.dat` file included with the package. 
If a data point does not fall within any of these sector intervals, `normalise_counts()` silently sets that point to 0.0. 
This script avoids such gaps by deriving the boundaries directly from the `TSTART` and `TSTOP` values ​​found in the FITS file headers.

Usage:
python build_tess_sectors.py "lc/*.fits"
python build_tess_sectors.py "lc/*.fits" --base tess_sectors.dat
python build_tess_sectors.py "lc/*.fits" --inplace   # updates the installed package

If the `--base` argument is provided, the existing entries in that file are preserved; sectors extracted from the FITS files are then either overwritten or appended to the list.
"""
import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits

BJD_REF = 2457000.0
PAD = 0.02  


def read_fits_sector(path):
    """Returns (sector, tstart, tstop) from a TESS lightcurve FITS file."""
    with fits.open(path) as hdul:
        hdr0, hdr1 = hdul[0].header, hdul[1].header
        sector = hdr0.get("SECTOR", hdr1.get("SECTOR"))
        tstart, tstop = hdr1.get("TSTART"), hdr1.get("TSTOP")
        if tstart is None or tstop is None:
            t = hdul[1].data["TIME"]
            t = t[np.isfinite(t)]
            if len(t) == 0:
                return None
            tstart, tstop = float(t.min()), float(t.max())
    if sector is None:
        return None
    return int(sector), float(tstart), float(tstop)


def load_base(path):
    """Reads the existing tess_sectors.dat file as {sector: (jd1, jd2)}."""
    rows = {}
    if not path or not os.path.exists(path):
        return rows
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            rows[int(parts[0])] = (float(parts[2]), float(parts[3]))
        except ValueError:
            continue
    return rows


def jd_to_ddmmyyyy(jd):
    """JD -> dd/mm/yyyy."""
    z = int(jd + 0.5)
    f = jd + 0.5 - z
    if z >= 2299161:
        alpha = int((z - 1867216.25) / 36524.25)
        z += 1 + alpha - alpha // 4
    b = z + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = int(b - d - int(30.6001 * e) + f)
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    return f"{day:02d}/{month:02d}/{year:04d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="for FITS files glob pattern")
    ap.add_argument("--base", default=None,
                    help="preserve the current tess_sectors.dat")
    ap.add_argument("-o", "--out", default="tess_sectors.dat")
    ap.add_argument("--inplace", action="store_true",
                    help="Update the file in the installed ECLIPSR package.")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        sys.exit(f"No files matching with '{args.pattern}'")

    found = {}
    for fn in files:
        try:
            res = read_fits_sector(fn)
        except Exception as exc:
            print(f"  Skipped {os.path.basename(fn)}: {exc}")
            continue
        if res is None:
            print(f"  Skipped {os.path.basename(fn)}: SECTOR/TIME can't be read")
            continue
        s, t0, t1 = res
        if s in found: 
            t0 = min(t0, found[s][0])
            t1 = max(t1, found[s][1])
        found[s] = (t0, t1)

    print(f"{len(files)} files read, {len(found)} sectors are found: "
          f"{sorted(found)}")

    rows = load_base(args.base)
    if rows:
        print(f"{len(rows)} rows were retrieved from the base file.")

    for s, (t0, t1) in found.items():
        rows[s] = (t0 + BJD_REF - PAD, t1 + BJD_REF + PAD)

    # Clean up cross-sector overlap: If one sector crosses the beginning of another, cut it off midway.
    keys = sorted(rows)
    for a, b in zip(keys[:-1], keys[1:]):
        if rows[a][1] > rows[b][0]:
            mid = 0.5 * (rows[a][1] + rows[b][0])
            rows[a] = (rows[a][0], mid)
            rows[b] = (mid, rows[b][1])
            print(f"  Sector {a}/{b} overlap was cut at point {mid:.4f}")

    # final check: each row must be internally consistent, and the chronological
    # order must be maintained. If not, do not write—a corrupted file is worse
    # than silently discarding the data.
    keys = sorted(rows)
    problems = []
    for s in keys:
        if rows[s][1] <= rows[s][0]:
            problems.append(f"  Sector {s}: end <= start "
                            f"({rows[s][0]:.3f} -> {rows[s][1]:.3f})")
    for a, b in zip(keys[:-1], keys[1:]):
        if rows[b][0] < rows[a][1]:
            problems.append(f"  Sector {a} and {b} are still in match")
    if problems:
        print("\nFILE NOT WRITTEN -- inconsistent lines:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        sys.exit("The times from the FITS file do not match the master table."
                 "The TSTART/TSTOP values ​​of the files and the --base file. "
                 "check (BTJD or full BJD?).")

    out = args.out
    if args.inplace:
        import eclipsr
        out = os.path.join(os.path.dirname(eclipsr.__file__),
                           "data", "tess_sectors.dat")
        print(f"file to write: {out}")

    with open(out, "w") as f:
        f.write("# sector\tdates\t\t\t    JD1\t\t     JD2\n")
        for s in sorted(rows):
            j1, j2 = rows[s]
            f.write(f"{s} {jd_to_ddmmyyyy(j1)}-{jd_to_ddmmyyyy(j2)}"
                    f"       {j1:.5f}       {j2:.5f}\n")
    print(f"{len(rows)} rows written -> {out}")


if __name__ == "__main__":
    main()