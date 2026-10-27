#!/usr/bin/env python3
"""Measure time and disk per target, then extrapolate to the full sample.

    python3 measure_run.py --list test/download_list.csv --n 60

Downloads a small stratified sample (so every pipeline is represented),
times it, measures the bytes it put on disk, and scales the result up using
the pipeline mix of the whole list.  Far cheaper than downloading 5000
targets to find out how much room they need.
"""

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd

from get_lightcurve import Download

MB = 1024 ** 2
GB = 1024 ** 3


def dir_bytes(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def main(list_file, n, out_dir, batch_size, keep):
    full = pd.read_csv(list_file)
    pairs = full[['tic', 'prov']].drop_duplicates()

    print(f'list: {len(full)} sector-rows, {len(pairs)} targets')
    print('\nrows per pipeline in the full list:')
    print(full['prov'].value_counts().to_string())

    # stratified sample: a share of each pipeline, at least 5 where possible
    per = max(5, n // max(full['prov'].nunique(), 1))
    sample = pairs.groupby('prov', group_keys=False).head(per).head(n)
    print(f'\nsampling {len(sample)} targets:')
    print(sample['prov'].value_counts().to_string())

    work = Path(out_dir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    sub = full.merge(sample, on=['tic', 'prov'], how='inner')
    sub_list = work / 'list.csv'
    sub.to_csv(sub_list, index=False)

    d = Download(sub_list, out_dir=work)

    t0 = time.perf_counter()
    d.resolve_products(batch_size=batch_size)
    t_resolve = time.perf_counter() - t0
    print(f'\nresolve_products: {t_resolve:.1f}s for '
          f'{sub["obsid"].nunique()} observations')

    rows = []
    for tic, prov in sample[['tic', 'prov']].values.tolist():
        before = dir_bytes(work)
        t0 = time.perf_counter()
        verdict = d.download_target(int(tic), prov)
        dt = time.perf_counter() - t0
        grew = dir_bytes(work) - before

        n_sec = int((sub['tic'] == tic).sum())
        rows.append({'tic': tic, 'prov': prov, 'verdict': verdict,
                     'sectors': n_sec, 'seconds': dt, 'bytes': grew})

    res = pd.DataFrame(rows)
    ok = res[res['verdict'].isin(['ok', 'partial'])]

    print('\n=== per pipeline, measured ===')
    stats = (ok.groupby('prov')
               .agg(targets=('tic', 'size'),
                    sectors=('sectors', 'sum'),
                    sec_per_target=('seconds', 'mean'),
                    mb_per_sector=('bytes', lambda s: 0.0))
               .reset_index())
    # bytes per sector, computed properly
    per_sec = (ok.groupby('prov')
                 .apply(lambda g: g['bytes'].sum() / max(g['sectors'].sum(), 1)))
    stats['mb_per_sector'] = stats['prov'].map(per_sec) / MB
    stats['sec_per_target'] = stats['sec_per_target'].round(2)
    stats['mb_per_sector'] = stats['mb_per_sector'].round(3)
    print(stats.to_string(index=False))

    failed = res[~res['verdict'].isin(['ok', 'partial'])]
    if len(failed):
        print(f'\n{len(failed)} sampled targets returned '
              f'{failed["verdict"].value_counts().to_dict()}')

    # ---- extrapolate with the pipeline mix of the whole list ----
    mix_rows = full['prov'].value_counts()                  # sector-rows
    mix_targets = pairs['prov'].value_counts()              # targets

    print('\n=== extrapolated to this whole list ===')
    total_gb = total_s = 0.0
    for prov in mix_rows.index:
        if prov not in set(stats['prov']):
            print(f'  {prov}: not sampled, skipped')
            continue
        r = stats.loc[stats['prov'] == prov].iloc[0]
        gb = mix_rows[prov] * r['mb_per_sector'] / 1024
        secs = mix_targets.get(prov, 0) * r['sec_per_target']
        total_gb += gb
        total_s += secs
        print(f'  {prov:5} {mix_targets.get(prov, 0):6} targets, '
              f'{mix_rows[prov]:6} sectors -> {gb:7.2f} GB, {secs / 60:7.1f} min')

    n_obs = full['obsid'].nunique()
    resolve_total = t_resolve * n_obs / max(sub['obsid'].nunique(), 1)
    print(f'\n  product lists           : {resolve_total / 60:7.1f} min')
    print(f'  transfers (serial)      : {total_s / 60:7.1f} min')
    print(f'  total (serial)          : {(total_s + resolve_total) / 60:7.1f} min '
          f'= {(total_s + resolve_total) / 3600:.1f} h')
    print(f'  disk                    : {total_gb:7.2f} GB')

    for workers in (4, 8):
        print(f'  transfers with {workers} threads : '
              f'{total_s / 60 / workers:7.1f} min')

    # ---- and to a bigger sample ----
    print('\n=== if the catalogue grows, same mix ===')
    for n_targets in (5000, 20000, 180000):
        scale = n_targets / len(pairs)
        print(f'  {n_targets:7} targets -> {total_gb * scale:8.1f} GB, '
              f'{(total_s + resolve_total) * scale / 3600:6.1f} h serial')

    if not keep:
        shutil.rmtree(work, ignore_errors=True)
        print(f'\nremoved {work}')
    else:
        print(f'\nsample left in {work} ({dir_bytes(work) / MB:.1f} MB)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--list', default='test_files/download_list.csv')
    p.add_argument('--n', type=int, default=60)
    p.add_argument('--out-dir', default='measure_sample')
    p.add_argument('--batch-size', type=int, default=200)
    p.add_argument('--keep', action='store_true',
                   help='do not delete the sampled files afterwards')
    a = p.parse_args()

    main(a.list, a.n, a.out_dir, a.batch_size, a.keep)