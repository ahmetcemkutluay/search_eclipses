#!/usr/bin/env python3
"""
tess_sectors.dat dosyasini kendi TESS FITS dosyalarindan uretir/gunceller.

Neden: ECLIPSR'in get_tess_sectors() fonksiyonu paketle gelen
eclipsr/data/tess_sectors.dat dosyasindaki JD sinirlarini kullanir. Bir veri
noktasi hicbir sektor araliginin icine dusmezse normalise_counts() o noktayi
sessizce 0.0 yapar. Bu script sinirlari dogrudan FITS basliklarindaki
TSTART/TSTOP degerlerinden aldigi icin boyle bir bosluk olusmaz.

Kullanim:
    python build_tess_sectors.py "lc/*.fits"
    python build_tess_sectors.py "lc/*.fits" --base tess_sectors.dat
    python build_tess_sectors.py "lc/*.fits" --inplace   # kurulu paketi gunceller

--base verilirse o dosyadaki satirlar korunur, FITS'ten gelen sektorler
onlarin uzerine yazilir / sonlarina eklenir.
"""
import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits

BJD_REF = 2457000.0
PAD = 0.02  # ~30 dakika emniyet payi (gun cinsinden)


def read_fits_sector(path):
    """Bir TESS lightcurve FITS dosyasindan (sektor, tstart, tstop) dondurur."""
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
    """Mevcut tess_sectors.dat dosyasini {sektor: (jd1, jd2)} olarak okur."""
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
    """JD -> dd/mm/yyyy (takvim tarihi, UT)."""
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
    ap.add_argument("pattern", help="FITS dosyalari icin glob deseni")
    ap.add_argument("--base", default=None,
                    help="korunacak mevcut tess_sectors.dat")
    ap.add_argument("-o", "--out", default="tess_sectors.dat")
    ap.add_argument("--inplace", action="store_true",
                    help="kurulu eclipsr paketindeki dosyayi guncelle")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        sys.exit(f"'{args.pattern}' ile eslesen dosya yok")

    found = {}
    for fn in files:
        try:
            res = read_fits_sector(fn)
        except Exception as exc:
            print(f"  atlandi {os.path.basename(fn)}: {exc}")
            continue
        if res is None:
            print(f"  atlandi {os.path.basename(fn)}: SECTOR/TIME okunamadi")
            continue
        s, t0, t1 = res
        if s in found:  # ayni sektorden birden fazla dosya -> birlestir
            t0 = min(t0, found[s][0])
            t1 = max(t1, found[s][1])
        found[s] = (t0, t1)

    print(f"{len(files)} dosya okundu, {len(found)} sektor bulundu: "
          f"{sorted(found)}")

    rows = load_base(args.base)
    if rows:
        print(f"temel dosyadan {len(rows)} satir alindi")

    for s, (t0, t1) in found.items():
        rows[s] = (t0 + BJD_REF - PAD, t1 + BJD_REF + PAD)

    # sektorler arasi ortusmeyi temizle: bir sektorun bitisi bir sonrakinin
    # baslangicini gecerse ikisinin ortasindan kes
    keys = sorted(rows)
    for a, b in zip(keys[:-1], keys[1:]):
        if rows[a][1] > rows[b][0]:
            mid = 0.5 * (rows[a][1] + rows[b][0])
            rows[a] = (rows[a][0], mid)
            rows[b] = (mid, rows[b][1])
            print(f"  sektor {a}/{b} ortusmesi {mid:.4f} noktasindan kesildi")

    # son kontrol: her satir kendi icinde tutarli ve zaman sirasi sektor
    # sirasiyla ayni olmali. Degilse yazma -- bozuk bir dosya sessizce
    # veriyi sifirlamaktan daha kotudur.
    keys = sorted(rows)
    problems = []
    for s in keys:
        if rows[s][1] <= rows[s][0]:
            problems.append(f"  sektor {s}: bitis <= baslangic "
                            f"({rows[s][0]:.3f} -> {rows[s][1]:.3f})")
    for a, b in zip(keys[:-1], keys[1:]):
        if rows[b][0] < rows[a][1]:
            problems.append(f"  sektor {a} ile {b} hala ortusuyor")
    if problems:
        print("\nDOSYA YAZILMADI -- tutarsiz satirlar:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        sys.exit("FITS'ten gelen zamanlar temel tabloyla uyusmuyor. "
                 "Dosyalarin TSTART/TSTOP degerlerini ve --base dosyasini "
                 "kontrol edin (BTJD mi, tam BJD mi?).")

    out = args.out
    if args.inplace:
        import eclipsr
        out = os.path.join(os.path.dirname(eclipsr.__file__),
                           "data", "tess_sectors.dat")
        print(f"yazilacak dosya: {out}")

    with open(out, "w") as f:
        f.write("# sector\tdates\t\t\t    JD1\t\t     JD2\n")
        for s in sorted(rows):
            j1, j2 = rows[s]
            f.write(f"{s} {jd_to_ddmmyyyy(j1)}-{jd_to_ddmmyyyy(j2)}"
                    f"       {j1:.5f}       {j2:.5f}\n")
    print(f"{len(rows)} satir yazildi -> {out}")


if __name__ == "__main__":
    main()