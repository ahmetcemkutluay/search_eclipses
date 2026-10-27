#!/usr/bin/env python3
"""Where do the SPOC targets go between 'has products' and 'assigned SPOC'?

    python3 why_few_spoc.py
"""

import pandas as pd

SUMMARY = 'test_files/tess_products_summary.csv'
LONG = 'test_files/tess_products_long.csv'
PROVS = ('SPOC', 'TGLC', 'QLP', 'T16')

info = pd.read_csv(SUMMARY)
prod = pd.read_csv(LONG)
prod['exptime'] = pd.to_numeric(prod['exptime'], errors='coerce')

n = len(info)
print(f'{n} targets in the catalogue\n')

print('=== targets with ANY product, by column ===')
for p in PROVS:
    has_prod = int((info.get(f'n_prod_{p}', pd.Series(0, index=info.index)) > 0).sum())
    has_sec = int((info.get(f'n_sec_{p}', pd.Series(0, index=info.index)) > 0).sum())
    print(f'  {p:5}  n_prod_{p} > 0 : {has_prod:7}   '
          f'n_sec_{p} > 0 : {has_sec:7}   lost: {has_prod - has_sec:7}')

print('\n  n_prod_* counts raw MAST observations (data-validation spans,')
print('  target-pixel-only observations and 20 s products included).')
print('  n_sec_*  counts sectors in the cleaned inventory.')

print('\n=== the funnel, pipeline by pipeline ===')
for p in PROVS:
    sub = prod[prod['prov'] == p]
    any_sec = sub['tic'].nunique()
    usable = sub[sub['exptime'] != 20]['tic'].nunique()
    print(f'  {p:5}  in inventory: {any_sec:7}   after dropping 20 s: {usable:7}')

print('\n=== why a target with SPOC data is not assigned SPOC ===')
spoc = set(prod.loc[(prod['prov'] == 'SPOC') & (prod['exptime'] != 20), 'tic'])
assigned = info.loc[info['prov'] == 'SPOC', 'tic'] if 'prov' in info.columns else []
print(f'  targets with a usable SPOC sector : {len(spoc)}')
print(f'  targets assigned SPOC             : {len(set(assigned))}')
print('  (these must match: SPOC is first in the priority order)')

only20 = (set(prod.loc[prod["prov"] == "SPOC", "tic"])
          - spoc)
print(f'\n  targets whose only SPOC products are 20 s: {len(only20)}')

print('\n=== magnitude: who has SPOC and who does not ===')
if 'tess_mag' in info.columns:
    has = info['tic'].isin(spoc)
    print(info.groupby(has)['tess_mag']
              .describe()[['count', 'mean', '50%', 'max']]
              .rename(index={False: 'no SPOC', True: 'has SPOC'}).to_string())

print('\n=== targets with nothing at all ===')
if 'prov' in info.columns:
    nan = info[info['prov'].isna()]
    print(f'  {len(nan)} targets ({100 * len(nan) / n:.1f}%)')
    if 'tess_mag' in info.columns:
        print(nan['tess_mag'].describe()[['count', 'mean', '50%', 'max']].to_string())
        bins = pd.cut(nan['tess_mag'], [0, 10, 12, 14, 16, 18, 30])
        print('\n  magnitude distribution:')
        print(bins.value_counts().sort_index().to_string())
    have_any = [c for c in info.columns if c.startswith('n_prod_')]
    if have_any:
        with_prod = int((nan[have_any].sum(axis=1) > 0).sum())
        print(f'\n  of these, {with_prod} DO have raw MAST products '
              f'(so they were lost to the 20 s / DV-span filters)')