"""Build the download list, then download it with N processes in parallel.

Stage 1 (once, in the parent process) runs the selection criteria over the
product inventory and writes download_list.csv.  Stage 2 shards that list by
tic % N, so no two processes ever touch the same tic_* folder or the same
status row.  Stop it whenever you like: each process writes its status on the
way out and picks up from there on the next run.  N = 1 runs it serially.
"""

from pathlib import Path
import argparse
import multiprocessing as mp
import time

import pandas as pd

from get_lightcurve import Criteria, Download

N = 6               # parallel processes
CHUNK = 2000        # targets resolved + transferred per round
BATCH = 200         # observations per get_product_list call
FLUSH = 200         # targets between status writes

# --- inputs and outputs ----------------------------------------------------

SUMMARY_FILE = Path('test_files') / 'tess_products_summary.csv'
LONG_FILE = Path('test_files') / 'tess_products_long.csv'
DOWNLOAD_LIST = Path('test_files') / 'download_list.csv'

LC_DIR = Path('test_files/lightcurves')
SHARD_DIR = Path('test_files/shards')
STATUS_DIR = Path('test_files/status')


def build_download_list(rebuild=False):
    """Run the selection criteria and write the download list.

    Called once, in the parent process, before any worker starts: the
    workers all read this file and must not race to write it.
    """
    if DOWNLOAD_LIST.is_file() and not rebuild:
        out = pd.read_csv(DOWNLOAD_LIST)
        print(f'{DOWNLOAD_LIST} exists: {len(out)} rows, '
              f'{out["tic"].nunique()} targets (--rebuild to redo it)')
        return out

    for f in (SUMMARY_FILE, LONG_FILE):
        if not f.is_file():
            raise SystemExit(f'{f} not found — run tess_products.py first')

    c = Criteria(SUMMARY_FILE, LONG_FILE)
    c.assign_all()

    DOWNLOAD_LIST.parent.mkdir(parents=True, exist_ok=True)
    out = c.write_output(DOWNLOAD_LIST)

    print(f'{DOWNLOAD_LIST}: {len(out)} rows, {out["tic"].nunique()} targets')
    print('\npipeline distribution:')
    print(c.info['prov'].value_counts(dropna=False).to_string())
    print('\ncadence distribution:')
    print(out['exptime'].value_counts(dropna=False).to_string())

    return out


def work(shard):
    SHARD_DIR.mkdir(exist_ok=True)
    STATUS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DOWNLOAD_LIST)
    df = df[df['tic'].astype(int) % N == shard]
    list_file = SHARD_DIR / f'download_list_{shard}.csv'
    df.to_csv(list_file, index=False)

    d = Download(list_file, out_dir=LC_DIR,
                 status_file=STATUS_DIR / f'download_status_{shard}.csv')

    pend = d.pending_rows()
    pairs = pend[['tic', 'prov']].drop_duplicates().values.tolist()
    print(f'[{shard}] {len(pend)} pending rows, {len(pairs)} targets to go',
          flush=True)

    # download_target() rewrites the whole status file after every target;
    # silence it and flush on our own schedule instead
    flush = d._save_status
    d._save_status = lambda: None

    t0 = time.perf_counter()
    n = 0
    try:
        for i in range(0, len(pairs), CHUNK):
            block = pairs[i:i + CHUNK]
            sel = pend.merge(pd.DataFrame(block, columns=['tic', 'prov']),
                             on=['tic', 'prov'], how='inner')
            ids = [o for o in sel['obsid'].unique() if str(o) not in ('', 'nan')]

            try:
                d.resolve_products(obsids=ids, batch_size=BATCH)
            except Exception as err:
                print(f'[{shard}] resolve failed, chunk skipped: '
                      f'{type(err).__name__}: {err}', flush=True)
                continue

            for tic, prov in block:
                try:
                    d.download_target(int(tic), prov)
                except Exception as err:
                    d._mark(int(tic), prov,
                            d.pending_sectors(int(tic), prov), 'error')
                    print(f'[{shard}] TIC {tic} failed: '
                          f'{type(err).__name__}: {err}', flush=True)
                n += 1
                if n % FLUSH == 0:
                    flush()

            flush()
            d._products, d._index = None, None

            el = time.perf_counter() - t0
            print(f'[{shard}] {n}/{len(pairs)} targets, {el / 3600:.1f} h, '
                  f'eta {el / max(n, 1) * (len(pairs) - n) / 3600:.1f} h',
                  flush=True)
    except KeyboardInterrupt:
        print(f'[{shard}] interrupted at {n}/{len(pairs)}', flush=True)
    finally:
        flush()

    return shard, n


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true',
                    help='rebuild download_list.csv even if it exists')
    ap.add_argument('--list-only', action='store_true',
                    help='write download_list.csv and stop')
    args = ap.parse_args()

    # ---- stage 1: the download list, once, before the pool ----
    build_download_list(rebuild=args.rebuild)

    if args.list_only:
        raise SystemExit(0)

    # ---- stage 2: download it, N processes in parallel ----
    t0 = time.perf_counter()
    try:
        with mp.Pool(N) as pool:
            pool.map(work, range(N))
    except KeyboardInterrupt:
        print('stopped')

    print(f'\n{(time.perf_counter() - t0) / 3600:.1f} h')

    files = sorted(STATUS_DIR.glob('download_status_*.csv'))
    if files:
        st = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f'{len(st)} sector-rows')
        print(st['status'].value_counts().to_string())