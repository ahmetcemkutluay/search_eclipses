"""Download the whole download_list.csv, N processes in parallel.

Each process takes the targets with tic % N == its own index, so no two
ever touch the same tic_* folder or the same status row.  Stop it whenever
you like: each process writes its status on the way out and picks up from
there on the next run.  N = 1 runs it serially.
"""

from pathlib import Path
import multiprocessing as mp
import time

import pandas as pd

from get_lightcurve import Download    

N = 6               # parallel processes
CHUNK = 2000        # targets resolved + transferred per round
BATCH = 200         # observations per get_product_list call
FLUSH = 200         # targets between status writes


def work(shard):
    Path('shards').mkdir(exist_ok=True)
    Path('status').mkdir(exist_ok=True)

    df = pd.read_csv('test/download_list.csv')
    df = df[df['tic'].astype(int) % N == shard]
    list_file = f'shards/download_list_{shard}.csv'
    df.to_csv(list_file, index=False)

    d = Download(list_file, out_dir='lightcurves',
                 status_file=f'status/download_status_{shard}.csv')

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
    t0 = time.perf_counter()
    try:
        with mp.Pool(N) as pool:
            pool.map(work, range(N))
    except KeyboardInterrupt:
        print('stopped')

    print(f'\n{(time.perf_counter() - t0) / 3600:.1f} h')

    files = sorted(Path('status').glob('download_status_*.csv'))
    if files:
        st = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f'{len(st)} sector-rows')
        print(st['status'].value_counts().to_string())