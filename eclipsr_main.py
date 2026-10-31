""" Run the ECLIPSR pipeline over the full catalogue, 10 processes.

    Same flow as the test block in do_eclipsr.py -- only the for-loop is replaced
    by a process pool, to speed up the process.
"""

import os

# Single-threaded BLAS. With N worker processes, each one spawning its own
# thread pool, the CPU is oversubscribed and the run ends up slower than with
# one thread per process. These must be set BEFORE numpy is imported, which is
# why this block sits above every other import.
# See: https://scikit-learn.org/stable/computing/parallelism.html
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import time
import signal
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

from do_eclipsr import ECLIPSR, Tee

path_records = Path('..') / 'platolops2_catalog' / 'records_v8.parquet'
path_status  = Path('.') / 'status' / 'download_status_all.csv'
path_lc      = Path('.') / 'lightcurves'
path_out     = Path('.') / 'results'


#number of processors, I use 14 processors as I have 24CPUs in Intel(R) Core(TM) i7-14650HX processor. Depending on your sample, you may use 4-6 processors.
N_PROC  =  14
SAVEFIG = True
TIMEOUT = 100

#--- worker: there should be an ECLIPSE in every process, at the module level ---
_E = None

class TargetTimeout(Exception):
    pass

def _on_alarm(signum, frame):
    raise TargetTimeout(f'exceeded {TIMEOUT}s')

def _init():
    global _E
    _E = ECLIPSR(path_records, path_status, path_lc, path_out, load_tables=False)
    if TIMEOUT:
        signal.signal(signal.SIGALRM, _on_alarm)


def _work(tic):
    """ The calculations do not write to the disk and do not throw errors--they
        return the result to the main process. A target that runs longer than
        TIMEOUT is abandoned and reported as failed, so one pathological light
        curve cannot block a worker for the rest of the run.
    """
    try:
        if TIMEOUT:
            signal.alarm(TIMEOUT)
        try:
            res = _E.eclipsr_analysis(tic, verbose=True, savefig=SAVEFIG)
        finally:
            if TIMEOUT:
                signal.alarm(0)   
        if res is None:
            return (tic, 'skipped', '', None, None)
        return (tic, 'ok', '', res['summary'], res['eclipses'])
    except Exception as err:
        return (tic, 'failed', f'{type(err).__name__}: {err}', None, None)


if __name__ == '__main__':

    log_file = path_out / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with Tee(log_file):
        start_time = time.time()
        print(f'log -> {log_file}')

        e = ECLIPSR(path_records, path_status, path_lc, path_out)

        e.reset_outputs() #each run, whole files are reseted

        tics = [t for t in e.tics.drop_duplicates()
                if (path_lc / ('tic_' + t)).is_dir()]
        print(f'{len(tics)} target | {e.tics.nunique() - len(tics)} without a file')
        print(f'{N_PROC} process, savefig={SAVEFIG}\n')

        ok, skipped, failed = [], [], []

        try:
            with Pool(N_PROC, initializer=_init) as pool:
                # chunksize = 1
                for i, (tic, status, reason, summary, ecl) in enumerate(
                        pool.imap_unordered(_work, tics, chunksize=1), 1):

                    if status == 'ok':
                        e.save_records({'summary': summary, 'eclipses': ecl})
                        ok.append(tic)
                    elif status == 'skipped':
                        skipped.append(tic)
                    else:
                        print(f'  FAILED {tic}: {reason}')
                        failed.append((tic, reason))

                    if i % 100 == 0 or i == len(tics):
                        el = time.time() - start_time
                        eta = (len(tics) - i) / (i / el) / 60
                        print(f'[{i}/{len(tics)}] ok {len(ok)} | skip {len(skipped)} '
                              f'| fail {len(failed)} | {i/el:.1f} tgt/s | ETA {eta:.0f} min')
        finally:
            e.save_failed(failed)

        print(f'\n{"="*60}')
        print(f'ok: {len(ok)} | skipped: {len(skipped)} | failed: {len(failed)}')

        if ok:
            res_df = pd.read_csv(path_out / 'results.csv', dtype={'tic': str})
            ecl_df = pd.read_csv(path_out / 'eclipses.csv', dtype={'tic': str})
            print(f'results.csv : {len(res_df)} row')
            print(f'eclipses.csv: {len(ecl_df)} row')

            cnt = (ecl_df.groupby('tic').size().rename('rows').to_frame()
                   .join(res_df.set_index('tic')['n_eclipses']))
            bad = cnt[cnt['rows'] != cnt['n_eclipses']]
            if len(bad):
                print(bad)

        print(f'Elapsed: {time.time() - start_time:.2f} seconds')